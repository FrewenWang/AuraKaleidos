QUANT_DIR=24126_16bits
compile(){

	rm -rf ${QUANT_DIR}
	mkdir ${QUANT_DIR}
	#
	qnn-onnx-converter \
		--input_network FaceDetection0114V4Main.onnx \
		--input_list input_list.txt \
		-o ${QUANT_DIR}/model_quant.cpp \
		#--act_bw 16  \
		#--use_per_channel_quantization
	
	qnn-model-lib-generator \
		-c ${QUANT_DIR}/model_quant.cpp \
		-b ${QUANT_DIR}/model_quant.bin \
		-o ${QUANT_DIR}/libs \
		-t x86_64-linux-clang aarch64-android
	
	qnn-net-run \
		--backend ${QNN_SDK_ROOT}/target/x86_64-linux-clang/lib/libQnnHtp.so \
		--model ${QUANT_DIR}/libs/x86_64-linux-clang/libmodel_quant.so \
		--input_list input_list.txt \
		--output_dir ${QUANT_DIR}/output \
		--debug
		
	#qnn-context-binary-generator \
	#	--backend ${QNN_SDK_ROOT}/target/x86_64-linux-clang/lib/libQnnHtp.so \
	#	--model ${QUANT_DIR}/libs/x86_64-linux-clang/libmodel_quant.so \
	#	--binary_file model.bin \
	#	--output_dir ${QUANT_DIR}
}

compilefp32(){ 
	rm -rf cpu_fp32
	mkdir cpu_fp32


	qnn-onnx-converter \
		--input_network FaceDetection0114V4Main.onnx \
		-o cpu_fp32/model_quant.cpp \
	
	qnn-model-lib-generator \
		-c cpu_fp32/model_quant.cpp \
		-b cpu_fp32/model_quant.bin \
		-o cpu_fp32/libs \
		-t x86_64-linux-clang
		
	qnn-net-run \
		--backend ${QNN_SDK_ROOT}/target/x86_64-linux-clang/lib/libQnnCpu.so \
		--model cpu_fp32/libs/x86_64-linux-clang/libmodel_quant.so \
		--input_list input_list.txt \
		--output_dir cpu_fp32/output_quant	\
		--debug
	

}

DEVICE_ID=79cf45bf
PROJECT_PATH_ON_DEVICE=/data/local/tmp/ruzhongl/debug_model

push_qnn(){

	adb -s ${DEVICE_ID} remount 
	
	adb -s ${DEVICE_ID} shell rm -rf ${PROJECT_PATH_ON_DEVICE}
	adb -s ${DEVICE_ID} shell mkdir -p ${PROJECT_PATH_ON_DEVICE}
	adb -s ${DEVICE_ID} shell mkdir -p ${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0
	adb -s ${DEVICE_ID} shell mkdir -p ${PROJECT_PATH_ON_DEVICE}/dsp

	
	adb -s ${DEVICE_ID} push ${QNN_SDK_ROOT}/target/aarch64-android/bin/qnn-net-run ${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0/
	adb -s ${DEVICE_ID} push ${QNN_SDK_ROOT}/target/aarch64-android/lib/libQnnHtpStub.so ${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0/
	adb -s ${DEVICE_ID} push ${QNN_SDK_ROOT}/target/aarch64-android/lib/libQnnGpu.so ${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0/
	adb -s ${DEVICE_ID} push ${QNN_SDK_ROOT}/target/hexagon-v68/lib/unsigned/* ${PROJECT_PATH_ON_DEVICE}/dsp/
	
}
push_files(){
	adb -s ${DEVICE_ID} push QnnHtpDebug.conf /data/local/tmp
	adb -s ${DEVICE_ID} push QnnHtpDebug.conf ${PROJECT_PATH_ON_DEVICE}/dsp/
	
	adb -s ${DEVICE_ID} push input_list.txt ${PROJECT_PATH_ON_DEVICE}
	adb -s ${DEVICE_ID}	push data.raw ${PROJECT_PATH_ON_DEVICE}
	
	#for htp
	adb -s ${DEVICE_ID} push ${QUANT_DIR}/model.bin.bin ${PROJECT_PATH_ON_DEVICE}
	
	#for gpu
	#adb -s ${DEVICE_ID} push cpu_fp32/libs_quant/aarch64-android/* ${PROJECT_PATH_ON_DEVICE}
}

run_on_device(){
	adb -s ${DEVICE_ID} logcat -c
	adb -s ${DEVICE_ID} shell "cd ${PROJECT_PATH_ON_DEVICE} &&
			   export ADSP_LIBRARY_PATH=\"/vendor/lib/rfsa/adsp;/vendor/lib/rfsa/dsp/sdk;/vendor/lib/rfsa/dsp/testsig;${PROJECT_PATH_ON_DEVICE}/dsp\" &&
			   export LD_LIBRARY_PATH=${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0:/vendor/lib64/ &&
			   export PATH=${PATH}:${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0 &&
			   ${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0/qnn-net-run \
					--backend  aarch64-android-clang6.0/libQnnHtpStub.so \
					--retrieve_context model.bin.bin \
					--profiling_level basic \
					--input_list input_list.txt \
					--perf_profile burst
			"
	adb -s ${DEVICE_ID} pull ${PROJECT_PATH_ON_DEVICE}/output ${QUANT_DIR}/output
	
	qnn-profile-viewer --input_log ${QUANT_DIR}/output/output/qnn-profiling-data.log --output ${QUANT_DIR}/output/output.csv
	#adb -s ${DEVICE_ID} logcat > logcat.log
}

run_on_device_gpu(){
	
	adb -s ${DEVICE_ID} logcat -c
	adb -s ${DEVICE_ID} shell "cd ${PROJECT_PATH_ON_DEVICE} &&
			   export ADSP_LIBRARY_PATH=\"/vendor/lib/rfsa/adsp;/vendor/lib/rfsa/dsp/sdk;/vendor/lib/rfsa/dsp/testsig;${PROJECT_PATH_ON_DEVICE}/dsp\" &&
			   export LD_LIBRARY_PATH=${PROJECT_PATH_ON_DEVICE}:${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0:/vendor/lib64/ &&
			   export PATH=${PATH}:${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0 &&
			   ${PROJECT_PATH_ON_DEVICE}/aarch64-android-clang6.0/qnn-net-run \
					--backend  aarch64-android-clang6.0/libQnnGpu.so \
					--model ${PROJECT_PATH_ON_DEVICE}/libmodel_quant.so \
					--profiling_level basic \
					--input_list input_list.txt \
					--perf_profile power_saver
			"
	adb -s ${DEVICE_ID} pull ${PROJECT_PATH_ON_DEVICE}/output cpu_fp32/output
	
	qnn-profile-viewer --input_log cpu_fp32/output/qnn-profiling-data.log --output cpu_fp32/output/output.csv
	adb -s ${DEVICE_ID} logcat > logcat.log
}

compile
compilefp32
#push_qnn
#push_files
#run_on_device
#run_on_device_gpu


