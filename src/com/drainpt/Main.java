package com.drainpt;

import java.awt.image.BufferedImage;

import com.drainpt.objects.Camera;
import com.drainpt.objects.Scene;

public class Main {

	// Entrypoint
	public static void main(String[] args) {
		
		BufferedImage outputImage = new BufferedImage(256, 256, BufferedImage.TYPE_INT_RGB);
		Pathtracer pt = new Pathtracer(new Scene(), new Camera(), outputImage);
		pt.close();
		
	}
	
}