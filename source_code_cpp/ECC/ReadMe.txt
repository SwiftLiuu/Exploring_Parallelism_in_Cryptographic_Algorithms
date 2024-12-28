
For serialCPU: I use visual studio build project, import mod.h and serial.cpp, put a input.txt into project folder
		In the Solution Explorer, right-click your project name and select Properties.
		In the Properties window, expand the Debugging section on the left-hand menu
		Set Command-Line Arguments: -t enc -b 7 input.txt decrypted_file.bin
		to let program run ECC encryption for input.txt with 7 bytes per data block and save in decrypted_file.bin
		Set Command-Line Arguments: -t dec -b 7 decrypted_file.bin output.txt
		to let program run ECC decryption for decrypted_file.bin with 7 bytes per data block and save in output.txt

For parallelCPU: I use visual studio build project, import mod.h and cpp file, put a input.txt into project folder
		In the Solution Explorer, right-click your project name and select Properties.
		In the Properties window, expand the Debugging section on the left-hand menu
		Set Command-Line Arguments: -t enc -b 7 input.txt decrypted_file.bin 40
		to let program run ECC encryption for input.txt with 7 bytes per data block and save in decrypted_file.bin, use 40 threads
		Set Command-Line Arguments: -t dec -b 7 decrypted_file.bin output.txt 40
		to let program run ECC decryption for decrypted_file.bin with 7 bytes per data block and save in output.txt, use 40 threads		

For parallelGPU: nvcc -O3 -o ecc.exe 123.cu to build program, put a large_input.txt into local folder
		in terminal cd to file, then run 
		.\ecc.exe -t enc -b 7 large_input.txt decrypted_file.bin 
		to let program run ECC encryption for large_input.txt with 7 bytes per data block and save in decrypted_file.bin
		in terminal run 
		.\ecc.exe -t dec -b 7 decrypted_file.bin output.txt
		to let program run ECC decryption for decrypted_file.bin with 7 bytes per data block and save in output.txt

I build three project on Windows, if  	
			