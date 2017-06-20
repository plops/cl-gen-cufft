(eval-when (:compile-toplevel :execute :load-toplevel)
  (ql:quickload :cl-cpp-generator))

(in-package :cl-cpp-generator)

(defmacro e (&body body)
  `(statements (<< "std::cout" ,@(loop for e in body collect
				      (cond ((stringp e) `(string ,e))
					    (t e))) "std::endl")))

(defmacro er (&body body)
  `(statements (<< "std::cerr" ,@(loop for e in body collect
				      (cond ((stringp e) `(string ,e))
					    (t e))) "std::endl")))

(defun replace-all (string part replacement &key (test #'char=))
"Returns a new string in which all the occurences of the part 
is replaced with replacement."
    (with-output-to-string (out)
      (loop with part-length = (length part)
            for old-pos = 0 then (+ pos part-length)
            for pos = (search part string
                              :start2 old-pos
                              :test test)
            do (write-string string out
                             :start old-pos
                             :end (or pos (length string)))
            when pos do (write-string replacement out)
            while pos))) 


(defparameter *trace-facts*
  `())

(let ((code `(with-compilation-unit
		 (include <array>)
	       (include <algorithm>)
	       (include <sstream>)
	       (include <fstream>)
	       (include <iostream>)
	       (include <cmath>)
		 	       
	       ;; self-contained interface to cufft
		 ;;https://developer.nvidia.com/cufft
		 ;;https://devblogs.nvidia.com/parallelforall/cuda-pro-tip-use-cufft-callbacks-custom-data-processing/
		 ;;https://github.com/drufat/cuda-examples/blob/master/cuda/fft.cu
		 (include <cufft.h>)
	       (include <cuda.h>)
	       (include <cuda_runtime_api.h>)
	       (raw ,(format nil "#define FatalError(s) ~a"
			     (replace-all
			      (emit-cpp
			       :code
			       `(do-while 0
				  (let ((_message :type "std::stringstream"))
				    (<< _message __FILE__ (char #\:)
					__LINE__
                                        ;(char #\:) __FUNCTION__
					(char #\Space)
					(funcall "std::string" s) "std::endl")
				    (<< "std::cerr" (funcall _message.str))
				    (funcall cudaDeviceReset)
                                        ;(funcall exit 1)
				    )))
			      "
"
			      "\\
")))

	       (raw ,(format nil "#define checkCuda(status) ~a"
			     (replace-all
			      (emit-cpp
			       :code
			       `(do-while 0
				  (let ((_error :type "std::stringstream"))
				    (if (!= 0 status)
					(statements
					 (<< _error (string " Cuda failure: ") (funcall cudaGetErrorString status))
					 (funcall FatalError (funcall _error.str)))))))
			      "
"
			      "\\
")))
	       (raw ,(format nil "#define checkCufft(status) ~a"
			     (replace-all
			      (emit-cpp
			       :code
			       `(do-while 0
				  (let ((_error :type "std::stringstream"))
				    (if (!= CUFFT_SUCCESS status)
					(statements
					 (<< _error (string "CUFFT failure: ") (funcall cufft_get_error_string status))
					 (funcall FatalError (funcall _error.str)))))))
			      "
"
			      "\\
")))
	       (raw "// float data[NZ][NY][NX]")
	       (raw "// float* flattened = data")
	       (raw "// data[z][y][x] == flattened[x+NX*y+NX*NY*z]")
	       (raw "// array<array<array<float,x>,y>,z> data_")
	       (raw "// float* flattened = data_")
	       (raw "// data_.at(x).at(y).at(z) == flattened[x+NX*y+NX*NY*z]")

	       (raw ,(format nil "#define ft_idx(x,y,z) ~a"
			     (replace-all
			      (emit-cpp
			       :code
			       `(+ (* z)
				   (* y NZ)
				   (* x NZ NY)))
			      "
"
			      "\\
")))
	       (raw ,(format nil "#define ft_idx2(x,y) ~a"
			     (replace-all
			      (emit-cpp
			       :code
			       `(+ (* x)
				   (* y NX)
				   ))
			      "
"
			      "\\
")))

	       (enum ft_constants_t 
		     (NX 256)
		     (NY 256)
		     (NZ 256))
	       (decl ((ft_data :type cufftComplex*)
		      (ft_plan :type cufftHandle)))
	       (function (cufft_get_error_string ((result :type cufftResult_t)) "const char*")
			 ,(let ((errs '((SUCCESS         #x0)
					(INVALID_PLAN    #x1)
					(ALLOC_FAILED    #x2)
					(INVALID_TYPE    #x3)
					(INVALID_VALUE   #x4)
					(INTERNAL_ERROR  #x5)
					(EXEC_FAILED     #x6)
					(SETUP_FAILED    #x7)
					(INVALID_SIZE    #x8)
					(UNALIGNED_DATA  #x9)
					(INCOMPLETE_PARAMETER_LIST  #xA)
					(INVALID_DEVICE  #xB)
					(PARSE_ERROR  #xC)
					(NO_WORKSPACE  #xD)
					(NOT_IMPLEMENTED  #xE)
					(LICENSE_ERROR  #x0F)
					(NOT_SUPPORTED  #x10))))
			       `(let ((msg :type ,(format nil "const std::array<const char*,~a>" (length errs)) :ctor (list (list ,@ (loop for (e f) in errs and i from 0 appending
																	  (progn
																	    (assert (= i f))
																	    `((string ,(format nil "~a" e)))))))))
				  (return (funcall msg.at result)))))
	       (function (ft_init () void)
			 (funcall checkCuda
				  (funcall cudaMallocManaged (funcall reinterpret_cast<void**> &ft_data) (* (funcall sizeof *ft_data) NX NY NZ) cudaMemAttachGlobal))
			 (funcall checkCufft
				  (funcall cufftPlan3d &ft_plan NX NY NZ CUFFT_C2C)))
	       (function (ft_fill_sinc ((data :type cufftComplex*)
					(radius :type float :default 1s0)) void)
			 ;; https://doi.org/10.1093/qmath/12.1.165 
			 (dotimes (k NZ)
			   (dotimes (j NY)
			     (dotimes (i NX)
			       (let ((x :ctor (+ -.5s0 (/ (funcall static_cast<float> i)
							  NX)))
				     (y :ctor (+ -.5s0 (/ (funcall static_cast<float> j)
							  NY)))
				     (z :ctor (+ -.5s0 (/ (funcall static_cast<float> k)
							  NZ)))
				     (r :ctor (* 2 (funcall static_cast<float> M_PI) radius (funcall "std::sqrt" (+ (* x x)
														    (* y y)
														    (* z z)))))
				     (sign :ctor (+ -1 (* 2 (% (+ i j k) 2)))
					   )
				     (alpha :ctor .54)
				     (beta :ctor (- 1 alpha))
				     (hamming_x :ctor (- alpha (* beta (funcall "std::cos" (funcall static_cast<float> (/ (* 2 M_PI i)
															  (- NX 1)))))))
				     (hamming_y :ctor (- alpha (* beta (funcall "std::cos" (funcall static_cast<float> (/ (* 2 M_PI j)
															  (- NY 1)))))))
				     (hamming_z :ctor (- alpha (* beta (funcall "std::cos" (funcall static_cast<float> (/ (* 2 M_PI k)
															  (- NZ 1)))))))
				     (hamming :ctor (* hamming_x
						       hamming_y
						       hamming_z)))
				 (if (== 0.0 r)
				     (setf (aref data (funcall ft_idx i j k)) (list (* sign 1s0) 0s0)) ;; make_cuComplex
				     (setf (aref data (funcall ft_idx i j k)) (list (/ (* hamming sign (funcall "std::sin" r))
										       r)
										    0s0))))))))
	       (function (ft_delete () void)
			 (funcall checkCufft
				  (funcall cufftDestroy ft_plan))
			 (funcall checkCuda
				  (funcall cudaFree ft_data)))

	       (function (pgm_write_xy ((fn :type "std::string")
					(data :type cufftComplex*)
					(z0 :type size_t)
					(scale :type float :default 0s0)
					) void
					  )
			 (let ((f :type "std::ofstream" :ctor (comma-list
							       fn
							       (|\|| "std::ofstream::out"
								     "std::ofstream::binary"
								     "std::ofstream::trunc")))
			       
			       (bufu8 :type "unsigned char*"  :ctor (new (aref "unsigned char" (* NX NY)))))

			   (if (== 0.0 scale)
			       (statements ;; find scaling
				(let ((buff32 :type "std::array<float,NX*NY>"))
				  (dotimes (j NY)
				    (dotimes (i NX)
				      (let ((sign :ctor (+ -1 (* 2 (% (+ i j z0) 2)))
						  ))
					(setf (aref buff32 (funcall ft_idx2 i j)) (* (funcall cuCabsf
											      (aref ft_data (funcall ft_idx i j z0))
											      ))))))
				  (let ((minmax :ctor (funcall "std::minmax_element" (funcall buff32.begin) (funcall buff32.end)))
					(mi :ctor (aref minmax.first 0))
					(ma :ctor (aref minmax.second 0)))
				    (for-range ((e :type auto&) buff32)
					       (setf e (/ (* 255s0 (- e mi))
							  (- ma mi)))))
				  (dotimes (i (funcall buff32.size))
				    (setf (aref bufu8 i) (funcall static_cast<int> (aref buff32 i))))))
			       (statements
				(dotimes (j NY)
				  (dotimes (i NX)
				    (let ((sign :ctor (+ -1 (* 2 (% (+ i j z0) 2)))
						))
				      (setf (aref bufu8 (funcall ft_idx2 i j)) (funcall "std::min" 255 (funcall "std::max" 0 (funcall static_cast<int> (* scale sign (funcall cuCrealf (aref ft_data (funcall ft_idx i j z0)))))))))))))

			   
			   
			   (<< f (string "P5\\n") NX (string " ")  NY  (string "\\n255\\n"))
			   (funcall f.write (funcall reinterpret_cast<char*> bufu8) (* NX NY))))
	       	       
	       (function (main ((argc :type int)
				(argv :type char**)) int)
			 

			 (funcall ft_init)
			 (funcall ft_fill_sinc ft_data (funcall static_cast<float> 60s0))
			 (funcall checkCufft (funcall cufftExecC2C ft_plan ft_data ft_data CUFFT_FORWARD))
			 (funcall cudaDeviceSynchronize)
			 (funcall pgm_write_xy (string "/dev/shm/o.pgm") ft_data (/ NZ 2))
			 (funcall ft_delete)))))
  (write-source "stage/cl-gen-cufft/source/main" "cpp" code)
  #+nil (sb-ext:run-program "/bin/sh" '("/home/martin/stage/cl-gen-cufft/run.sh"))
  )


