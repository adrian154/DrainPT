package com.drainpt.objects;

// Container class
public class Vector {

	public float x;
	public float y;
	public float z;
	
	public Vector() {
		
	}
	
	public Vector(float x) {
		this.x = x;
		this.y = x;
		this.z = x;
	}
	
	public Vector(float x, float y, float z) {
		this.x = x;
		this.y = y;
		this.z = z;
	}
	
}
