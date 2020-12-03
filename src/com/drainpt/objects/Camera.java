package com.drainpt.objects;

// Container class
public class Camera {

	public Vector position;
	public float FOV;
	
	public Camera() {
		this(new Vector(), 60);
	}
	
	public Camera(Vector position, float FOV) {
		this.position = position;
		this.FOV = FOV;
	}
	
}
