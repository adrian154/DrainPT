package com.drainpt;

import java.awt.image.BufferedImage;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.Sizeof;
import org.jocl.cl_command_queue;
import org.jocl.cl_context;
import org.jocl.cl_device_id;
import org.jocl.cl_kernel;
import org.jocl.cl_mem;
import org.jocl.cl_platform_id;
import org.jocl.cl_program;
import org.jocl.cl_queue_properties;

import com.drainpt.objects.Camera;
import com.drainpt.objects.Scene;
import com.drainpt.util.CLHelper;

// This class is the most scuffed of all, so it warrants some explanation.

// - Initialize radiance buffer to 0, sample count to 0
// - For each sample:
//     - Run camera kernel (camera info -> rays [rayOriginBuf, rayDirBuf])
//         - This jitters the rays, so it has to be re-run every sample
//         - It also populates the primary raylist
//     - For each bounce:
//         - For every ray in the primary raylist, run raytrace kernel (rays -> hits [hitMtlIdxBuf, hitPointBuf, hitNormalBuf])
//             - Populates the secondary raylist
//             - Kernel atomically increments nextRayIdxBuf to append rays to the list
//         - For every ray in the primary raylist, run shading kernel (hits -> new rays, alter maskBuf, accumulatorBuf)
//             - Necessary since even missed rays need to be shaded with the sky
//         - Swap raylists so only rays that hit are extended
// - Divide radiance buffer by sample count
// - Release all buffers except radiance
// - Run tonemapping kernel (radiance -> final image)

public class Pathtracer {
	
	// OpenCL info
	protected cl_platform_id platformID;
	protected cl_device_id deviceID;
	
	// OpenCL objects
	protected cl_context context;
	protected cl_command_queue commandQueue;
	protected cl_program program;
	
	// Buffers
	protected List<cl_mem> buffers;
	
	// PRNG seed buffers
	protected cl_mem prngSeedsBuf;
	
	// ray origin buffers
	protected cl_mem rayOriginBuf;
	protected cl_mem rayDirBuf;
	
	// hit material buffers
	protected cl_mem hitMtlIdxBuf;
	protected cl_mem hitPointBuf;
	protected cl_mem hitNormalBuf;
	
	// underlying raylist buffers
	// DO NOT USE THESE DIRECTLY, USE `curRayList` AND `nextRayList`
	protected cl_mem rayList0Buf;
	protected cl_mem rayList1Buf;

	// teeny tiny cutesy little baby buffer, only one int
	// stores next index in raylists
	protected cl_mem numRaysBuf;
	
	// temporary buffers, reset per-sample
	protected cl_mem maskBuf;
	protected cl_mem accumulatorBuf;
	
	// slightly more long lasting buffer, stores accumulated radiance
	// reset per render cycle
	protected cl_mem radianceBuf;

	// helper for raylists
	// USE THESE INSTEAD OF THE UNDERLYING RAYLIST BUFFERS
	protected cl_mem curRayList;
	protected cl_mem nextRayList;
	
	// Object buffers
	protected cl_mem sphereCenterBuffer;
	protected cl_mem sphereRadiusBuffer;
	
	// Kernels
	protected cl_kernel generateRaysKernel;
	protected cl_kernel intersectKernel;
	
	// properties
	protected Scene scene;
	protected Camera camera;
	protected BufferedImage output;

	// transient state for kernel dispatch
	// this is absolutely terrible since the pathtracer is absolutely not reentrant but what can you do :)
	protected cl_kernel currentKernel;
	protected int currentArg;
	
	public Pathtracer(Scene scene, Camera camera, BufferedImage output) {
		
		this.scene = scene;
		this.camera = camera;
		this.output = output;
		
		initCompute();
		
	}
	
	private void initCompute() {
		
		// This is critical: it tells the CL bindings to use exceptions since we live in a civilized society
		// Avoids the need to check for errors at every turn
		CL.setExceptionsEnabled(true);
		
		selectPlatformAndDevice();
		setupCLObjects();
		
		try {
			setupCLProgram();
		} catch(IOException exception) {
			System.out.println("Failed to set up kernels: " + exception.getMessage());
		}
		
		initKernels();
		allocateBuffers();
		
	}
	
	private void selectPlatformAndDevice() {
		
		// Select arbitrary platform and device
		this.platformID = selectPlatform();
		this.deviceID = selectDevice();
				
		// Print some debug info
		System.out.printf("Using device \"%s\" on platform \"%s\"\n", CLHelper.getDeviceName(this.deviceID), CLHelper.getPlatformName(this.platformID));
		
	}
	
	private cl_platform_id selectPlatform() {
		return CLHelper.getPlatformIDs()[0];
	}
	
	private cl_device_id selectDevice() {
		return CLHelper.getDeviceIDs(this.platformID)[0];
	}
	
	private void setupCLObjects() {
		this.context = CLHelper.createContext(platformID, deviceID);
		this.commandQueue = CL.clCreateCommandQueueWithProperties(this.context, this.deviceID, new cl_queue_properties(), null);
	}
	
	private void setupCLProgram() throws IOException {
		String programSource = CLHelper.readFile("src/kernels/kernels.cl");
		this.program = CL.clCreateProgramWithSource(this.context, 1, new String[] {programSource}, null, null);
		CL.clBuildProgram(program, 0, null, null, null, null);		
	}
	
	private cl_mem createBuffer(long flags, long size, Pointer hostPtr) {
		cl_mem buffer = CL.clCreateBuffer(this.context, flags, size, hostPtr, null);
		buffers.add(buffer);
		return buffer;
	}
	
	private cl_mem createBuffer(long flags, long size) {
		return createBuffer(flags, size, null);
	}
	
	private void allocateBuffers() {
		this.buffers = new ArrayList<cl_mem>();
		this.numRaysBuf = createBuffer(CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, Sizeof.cl_int, Pointer.to(new int[] {0}));
		this.allocScreenBuffers();
		this.allocSceneBuffers();
		this.curRayList = rayList0Buf;
		this.nextRayList = rayList1Buf;
	}

	private void allocScreenBuffers() {
		int numPixels = output.getWidth() * output.getHeight();
		this.rayOriginBuf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_float3);
		this.rayDirBuf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_float3);
		this.hitMtlIdxBuf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_int);
		this.hitPointBuf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_float3);
		this.hitNormalBuf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_float3);
		this.rayList0Buf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_int);
		this.rayList1Buf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_int);
		this.maskBuf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_float3);
		this.accumulatorBuf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_float3);
		this.radianceBuf = createBuffer(CL.CL_MEM_READ_WRITE, numPixels * Sizeof.cl_float3);
		this.initPRNGSeedBuffer(numPixels);
	}
	
	private void allocSceneBuffers() {
		
		float[] centerBuf = scene.getSphereCenterBuffer();
		this.sphereCenterBuffer = createBuffer(CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, centerBuf.length * Sizeof.cl_float, Pointer.to(centerBuf));
		
		float[] radiusBuf = scene.getSphereRadiusBuffer();
		this.sphereRadiusBuffer = createBuffer(CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, radiusBuf.length * Sizeof.cl_float, Pointer.to(radiusBuf));
		
	}
	
	private void freeBuffers() {
		for(cl_mem buffer: buffers) {
			CL.clReleaseMemObject(buffer);
		}
	}
	
	private void initKernels() {
		this.generateRaysKernel = CL.clCreateKernel(program, "generateRaysKernel", null);
		this.intersectKernel = CL.clCreateKernel(program, "intersectKernel", null);
	}
	
	private void DISPATCH(long size) {
		
		CL.clEnqueueNDRangeKernel(
			this.commandQueue,
			this.currentKernel,
			1,
			null,
			new long[] {size},
			null,
			0,
			null,
			null
		);
		
		this.currentKernel = null;
		this.currentArg = 0;
		
	}
	
	private void KERNEL(cl_kernel kernel) {
		this.currentKernel = kernel;
	}
	
	private void ARGUMENT(int size, Pointer value) {
		CL.clSetKernelArg(this.currentKernel, this.currentArg++, size, value);
	}
	
	private void ARGUMENT(cl_mem buf) {
		this.ARGUMENT(Sizeof.cl_mem, Pointer.to(buf));
	}
	
	private void ARGUMENT(int val) {
		this.ARGUMENT(Sizeof.cl_int, Pointer.to(new int[] {val}));
	}
	
	private void ARGUMENT(float val) {
		this.ARGUMENT(Sizeof.cl_float, Pointer.to(new float[] {val}));
	}
	
	private void initPRNGSeedBuffer(int size) {

		Random random = new Random();
		long[] arr = new long[size];
		for(int i = 0; i < size; i++) {
			arr[i] = random.nextLong();
		}
		
		this.prngSeedsBuf = createBuffer(CL.CL_MEM_READ_WRITE | CL.CL_MEM_COPY_HOST_PTR, size * Sizeof.cl_long, Pointer.to(arr));
		
	}
	
	private void resetRadiances() {
		CL.clEnqueueFillBuffer(commandQueue, radianceBuf, Pointer.to(new float[] {0}), 4, 0, this.output.getWidth() * this.output.getHeight() * 4, 0, null, null);
	}
	
	private int getNumRays() {
		
		int[] arr = new int[1];
		CL.clEnqueueReadBuffer(commandQueue, numRaysBuf, true, 0, Sizeof.cl_int, Pointer.to(arr), 0, null, null);
		return arr[0];
		
	}
	
	private void setNumRays(int num) {
		CL.clEnqueueWriteBuffer(commandQueue, numRaysBuf, true, 0, 4, Pointer.to(new int[] {num}), 0, null, null);
	}
	
	private void generatePrimaryRays() {
		KERNEL(generateRaysKernel);
		ARGUMENT(this.rayOriginBuf);
		ARGUMENT(this.rayDirBuf);
		ARGUMENT(this.accumulatorBuf);
		ARGUMENT(this.maskBuf);
		ARGUMENT(this.curRayList);
		ARGUMENT(this.numRaysBuf);
		ARGUMENT(output.getWidth());
		ARGUMENT(output.getHeight());
		ARGUMENT(this.camera.position.x);
		ARGUMENT(this.camera.position.y);
		ARGUMENT(this.camera.position.z);
		ARGUMENT(this.camera.FOV);
		DISPATCH(output.getWidth() * output.getHeight());
	}
	
	private void flipRayLists() {
		cl_mem temp = this.nextRayList;
		this.curRayList = this.nextRayList;
		this.nextRayList = temp;
	}
	
	private void doIntersectionPass() {
		KERNEL(intersectKernel);
		ARGUMENT(this.rayOriginBuf);
		ARGUMENT(this.rayDirBuf);
		ARGUMENT(this.curRayList);
		ARGUMENT(this.hitPointBuf);
		ARGUMENT(this.hitNormalBuf);
		ARGUMENT(this.sphereRadiusBuffer);
		ARGUMENT(this.sphereRadiusBuffer);
		ARGUMENT(this.scene.getNumSpheres());
		ARGUMENT(this.rayList0Buf);
		flipRayLists();
	}
	
	public void render() {
		
		resetRadiances();
		
		int NUM_SAMPLES = 25;
		for(int sample = 0; sample < NUM_SAMPLES; sample++) {
			doSample();
		}
		
	}
	
	private void doSample() {
		
		generatePrimaryRays();
		
		int MAX_BOUNCES = 5;
		for(int i = 0; i < MAX_BOUNCES; i++) {
		
			this.doIntersectionPass();
			
		}
		
	}
	
	public void close() {
		freeBuffers();
	}
	
}