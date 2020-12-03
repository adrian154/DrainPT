package com.drainpt.util;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Paths;

import org.jocl.CL;
import org.jocl.Pointer;
import org.jocl.cl_context;
import org.jocl.cl_context_properties;
import org.jocl.cl_device_id;
import org.jocl.cl_platform_id;

// Helper class for commonly used OpenCL functions
public class CLHelper {

	// Read source so it can be compiled
	public static String readFile(String path) throws IOException {
		return new String(Files.readAllBytes(Paths.get(path)));
	}
	
	// Get number of platforms available
	public static int getNumPlatforms() {
		int[] numPlatformsArray = new int[1];
		CL.clGetPlatformIDs(0, null, numPlatformsArray);
		return numPlatformsArray[0];
	}
	
	// Get number of devices available
	public static int getNumDevices(cl_platform_id platform) {
		int[] numDevicesArray = new int[1];
		CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, 0, null, numDevicesArray);
		return numDevicesArray[0];
	}
	
	// Get list of platform IDs
	public static cl_platform_id[] getPlatformIDs() {
		int numPlatforms = CLHelper.getNumPlatforms();
		cl_platform_id[] platforms = new cl_platform_id[numPlatforms];
		CL.clGetPlatformIDs(numPlatforms, platforms, null);
		return platforms;
		
	}
	
	public static cl_device_id[] getDeviceIDs(cl_platform_id platform) {
		int numDevices = CLHelper.getNumDevices(platform);
		cl_device_id[] devices = new cl_device_id[numDevices];
		CL.clGetDeviceIDs(platform, CL.CL_DEVICE_TYPE_ALL, numDevices, devices, null);
		return devices;
		
	}
	
	public static String getPlatformName(cl_platform_id platform) {
		long[] platformNameSizeArray = new long[1];
		CL.clGetPlatformInfo(platform, CL.CL_PLATFORM_NAME, 0, null, platformNameSizeArray);
		byte[] buffer = new byte[(int)platformNameSizeArray[0]];
		CL.clGetPlatformInfo(platform, CL.CL_PLATFORM_NAME, buffer.length, Pointer.to(buffer), null);
		return new String(buffer, 0, buffer.length - 1);
		
	}
	
	public static String getDeviceName(cl_device_id device) {
		long[] deviceNameSizeArray = new long[1];
		CL.clGetDeviceInfo(device, CL.CL_DEVICE_NAME, 0, null, deviceNameSizeArray);
		byte[] buffer = new byte[(int)deviceNameSizeArray[0]];
		CL.clGetDeviceInfo(device, CL.CL_DEVICE_NAME, buffer.length, Pointer.to(buffer), null);
		return new String(buffer, 0, buffer.length - 1);
		
	}
		
	public static cl_context createContext(cl_platform_id platform, cl_device_id device) {
		cl_context_properties contextProperties = new cl_context_properties();
		contextProperties.addProperty(CL.CL_CONTEXT_PLATFORM, platform);
		cl_device_id[] deviceArray = new cl_device_id[] {device};
		cl_context context = CL.clCreateContext(contextProperties, 1, deviceArray, null, null, null);
		return context;
	}
	
	public static void listAllDevices() {
		cl_platform_id[] platforms = getPlatformIDs();
		for(cl_platform_id id: platforms) {
			System.out.println("Platform \"" + getPlatformName(id) + "\" has " + getNumDevices(id) + " device(s).");
			listDevices(id);
		}
	}
	
	public static void listDevices(cl_platform_id platform) {
		cl_device_id[] devices = getDeviceIDs(platform);
		int i = 0;
		for(cl_device_id id: devices) {
			System.out.println(i + ": " + getDeviceName(id));
			i++;
		}
	}
	
}
