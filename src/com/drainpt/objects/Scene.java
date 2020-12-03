package com.drainpt.objects;

import java.util.ArrayList;
import java.util.List;

public class Scene {

	private List<Sphere> spheres;
	
	public Scene() {
		spheres = new ArrayList<Sphere>();
	}
	
	public void add(Sphere sphere) {
		spheres.add(sphere);
	}
	
	public float[] getSphereCenterBuffer() {
		
		float[] buf = new float[spheres.size() * 4];
		
		for(int i = 0; i < spheres.size(); i++) {
			Sphere sphere = spheres.get(i);
			buf[i * 4] = sphere.center.x;
			buf[i * 4 + 1] = sphere.center.y;
			buf[i * 4 + 2] = sphere.center.z;
		}
		
		return buf;
		
	}
	
	public float[] getSphereRadiusBuffer() {
		
		float[] buf = new float[spheres.size()];
		
		for(int i = 0; i < spheres.size(); i++) {
			Sphere sphere = spheres.get(i);
			buf[i] = sphere.radius;
		}
		
		return buf;
		
	}
	
	public int getNumSpheres() {
		return this.spheres.size();
	}
	
}
