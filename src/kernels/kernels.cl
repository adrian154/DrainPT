// ----- Info
// CONVENTIONS:
// * Normals returned by intersection functions may not be normalized.
//     * They are normalized in the raytracing kernel once a minimal intersection is confirmed.
//     * This mitigates the need for expensive square roots.

// ----- Structures
struct Hit {
	bool hit;
	float3 point;
	float3 normal;
	float distance;
};

// ----- Helper functions
// Feel no shame in using these as they are inlined anyways :)

// ----- Intersection functions

// The derivation for this is pretty tedious if you aren't clever about the vectors
// However, it's pretty simple and yields a nice sensical quadratic
struct Hit intersectSphere(
	const float3 center,
	const float radius,
	float3 rayOrigin,
	float3 rayDirection
) {

	float3 origin = rayOrigin - center;
	float b = 2 * dot(origin, rayDirection);
	float c = dot(origin, origin) - radius * radius;
	
	struct Hit result;
	result.hit = false;
	
	float discrim = b * b - 4 * c;
	if(discrim < 0) {
		return result;
	}
	
	discrim = sqrt(discrim);
	float t = -b - discrim;
	if(t < 0) {
		t = -b + discrim;
		if(t < 0) {
			return result;
		}
	}
	
	result.hit = true;
	result.point = rayOrigin + rayDirection * t;
	result.normal = result.point - center;
	result.distance = t;
	return result;

}

// ----- Kernels

__kernel void generateRaysKernel(
	__global float3 *rayOriginsBuffer,
	__global float3 *rayDirectionsBuffer,
	__global float3 *accumulatorBuffer,
	__global float3 *maskBuffer,
	__global int *rayListBuffer,
	const int width,
	const int height,
	const float cameraX,
	const float cameraY,
	const float cameraZ,
	const float FOV
) {
	
	const int workItemID = get_global_id(0);
	
	int pixelX = workItemID % width;
	int pixelY = height - workItemID / width;
	
	// convert to image plane
	float x = (float)pixelX - (float)width / 2.0F;
	float y = (float)pixelY - (float)height / 2.0F;
	if(width > height) {
		x /= (float)width;
		y /= (float)width;
	} else {
		x /= (float)height;
		y /= (float)height;
	}
	
	// write ray
	float focalLength = 1 / tan(FOV * M_PI / 360);
	rayOriginsBuffer[workItemID] = (float3)(cameraX, cameraY, cameraZ);
	rayDirectionsBuffer[workItemID] = normalize((float3)(x, y, focalLength));
	
	// write mask/accumulator
	maskBuffer[workItemID] = (float3)(1.0f, 1.0f, 1.0f);
	accumulatorBuffer[workItemID] = (float3)(0.0f, 0.0f, 0.0f);
	
	// write raylist
	rayListBuffer[workItemID] = workItemID;
	
}

__kernel void intersectKernel(
	__global float3 *rayOriginsBuffer,
	__global float3 *rayDirectionsBuffer,
	__global int *rayListBuffer,
	__global int *hitMtlIdxBuffer,
	__global float3 *hitPointBuffer,
	__global float3 *hitNormalBuffer,
	__global float *sphereRadiusBuffer,
	__global float *sphereCenterBuffer,
	const int numSpheres,
) {

	const int workItemID = get_global_id(0);
	const int rayIndex = rayListBuffer[workItemID];

	float3 rayOrigin = rayOriginsBuffer[rayIndex];
	float3 rayDirection = rayDirectionsBuffer[rayIndex];
	
	struct Hit nearest;
	nearest.distance = INFINITY;
	for(int i = 0; i < numSpheres; i++) {
	
		float3 center = sphereCenterBuffer[i];
		float radius = sphereRadiusBuffer[i];
		struct Hit hit = intersectSphere(center, radius, rayOrigin, rayDirection);
		
		if(hit.hit && hit.distance < nearest.distance) {
			nearest = hit;
		}
		
	}
	
	if(nearest.hit) {
	
		// normalize normal
		nearest.normal = normalize(nearest.normal);
	
		// write to hitbuffer
		hitPointBuffer[rayIndex] = nearest.point;
		hitNormalBuffer[rayIndex] = nearest.normal;
		
		// TODO: Implement actual materials :)
		hitMtlIdxBuffer[rayIndex] = 0;
	
	} else {
	
		// indicate that the ray missed by writing a bogus value to the hit material index buffer
		hitMtlIdxBuffer[rayIndex] = -1;
	
	}

}

__kernel void shadeKernel(
	__global float3 *rayOrigins,
	__global float3 *rayDirectionsBuffer,
	__global int *hitMtlIdxBuffer,
	__global float3 *hitPointBuffer,
	__global float3 *hitNormalBuffer,
	__global float3 *maskBuffer,
	__global float3 *accumulatorBuffer,
	__global int *rayListBuffer,
	__global int *nextRayListBuffer
	__global int *rayIdx
) {
	
	const int workItemID = get_global_id(0);
	const int rayIndex = rayListBuffer[workItemID];
	
	int hitMtlIdx = hitMtlIdxBuffer[rayIndex];
	
	if(hitMtlIdx == -1) {
	
		// The ray missed
		// TODO: Sky shading
		accumulatorBuffer[rayIndex] = maskBuffer[rayIndex] * (float3)(1.0f, 1.0f, 1.0f);
		
	} else {
	
		// Get hit info
		float3 hitPoint = hitPointBuffer[rayIndex];
		float3 hitNormal = hitNormalBuffer[rayIndex];
		float3 incident = rayDirectionsBuffer[rayIndex];
		
		// Shade material
		
		// Write new ray to raybuffer
	
		// Add to raylist
		nextRayListBuffer[*rayIdx] = rayIndex;
		atomic_add(rayIdx, 1);
	
	}
	
}