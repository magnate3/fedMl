
```
class FedMLMqttInference:
```

```
inference_utils.model_partition or   inference model_partition or class ModelPartitioner or
model_partition AlexNet  ResNet LeNet   EdgeOrchestrator  EdgeDevice CloudDevice EdgeDevice  EdgeServer
"EdgeDevice EdgeServer "  "cloud-edge-device collaborative DNN inference" "Cloud-Edge Collaborative Inference"
"InferenceRequest" " asyncio.Queue 、asyncio.gather、 asyncio.get_event_loop 、 asyncio.Task.all_tasks"
"Neurosurgeon、Edgent和AdaComp"
```
【协同DNN推理】2017，DDNN，distributed DNN on device, edge and cloud   

[DADS端云协同](https://github.com/Tjyy-1223/DADS/tree/ddc709d75cf25ae30977fa888553114e17aa9260)    
[Tjyy-1223Neurosurgeon端云协同](https://github.com/Tjyy-1223/Neurosurgeon/tree/0f352d82a3b3e0951a5a72a5a7ea2aa605605d43)

[wslwy/Neurosurgeon](https://github.com/wslwy/Neurosurgeon/tree/main)    


[Autodidactic Neurosurgeon Collaborative Deep Inference for Mobile Edge Intelligence via Online Learning](https://github.com/letian-zhang/ANS/blob/main/client_camera_main.py)

[Edge-Intelligence/卷积神经网络协同推断仿真系统](https://github.com/wyc941012/Edge-Intelligence)   

[Shrink models、Distill sub-models、Ensemble](https://github.com/falcon-xu/DeViT)   

[On the impact of deep neural network calibration on adaptive edge offloading](https://github.com/pachecobeto95/early_exit_calibration/tree/main)   

[edge_api.py端云协同](https://github.com/wslwy/Cached-Infer/blob/57b0b0229cdbe335cc2e365dfc35228c01692f39/edge_api.py)   

[MatthiasJReisinger/addnn](https://github.com/MatthiasJReisinger/addnn/tree/main)   
[DNN云边协同工作汇总](https://github.com/Tjyy-1223/Collaborative-Inference-Work-Summary)   

[early_exit_calibration/appEdge/api/controllers.py](https://github.com/pachecobeto95/early_exit_calibration/blob/12efee86c7dc299a1bf56f6c2be1f32a14d952fa/appEdge/api/controllers.py)      


[adaptive-edge-computing-framework端云协同 ](https://github.com/cloudNativeAiops/adaptive-edge-computing-framework/blob/main/src/core/task_scheduler.py)   
  
[FedLLM: Build Your Own Large Language Models on Proprietary Data using the FedML Platform](https://github.com/ZongHR/FedGTS/tree/5520544f533d0e3f42ad8219c34e1c3cb54649af/python/spotlight_prj/fedllm)   

[python-p2p-network框架]()

[VATE: Edge-Cloud System for Object Detection in Real-Time Video Streams](https://github.com/m-maresch/vate/tree/main)


[VATEv2: Edge-Cloud System for Object Detection in Real-Time Video Streams](https://github.com/Gravarica/Vate-V2/tree/488d8be4904469f9bc00618813db88eff9465c0e)    

[VATE-Benchmark](https://github.com/Gravarica/VATE-Benchmark/blob/master/edge_device/detection.py)   


[联邦学习tungngreen/PipelineScheduler](https://github.com/tungngreen/PipelineScheduler)   

[modelparallel](https://github.com/tkazusa/smdistributed_modelparallel/tree/main/smdistributed/modelparallel/test/torch)  


[FlexNN](https://github.com/xxxxyu/FlexNN/blob/main/Evaluation.md)

+  cloud-edge-device inference  
```
Frameworks & Inference Engines 
Roboflow Inference (roboflow/inference): Designed to run computer vision models from the cloud to tiny edge devices (like Raspberry Pi, NVIDIA Jetson). It manages cameras, runs workflows, and optimizes performance across different hardware.
Triton Inference Server (triton-inference-server/server): NVIDIA's open-source, high-performance solution for optimized cloud and edge inferencing, standardizing model deployment across various workloads.
OpenVINO (openvinotoolkit/openvino): An open-source toolkit by Intel for optimizing and deploying AI inference on Intel hardware, often used in edge scenarios.
MNN (alibaba/MNN): A lightweight deep neural network inference engine for running models on devices, known for its blazing fast performance.
Tengine (OPEN AI LAB): An AI application development platform dedicated to AIoT scenarios, addressing fragmentation in the industry chain.
deepC: A vendor-independent deep learning library, compiler, and inference framework specifically for small form-factor devices and microcontrollers. 
On-Device LLM Resources
llama.cpp (ggerganov/llama.cpp): A lightweight C/C++ library for efficient Large Language Model (LLM) inference on various hardware, including edge devices.
Awesome LLMs on Device (NexaAI/Awesome-LLMs-on-device): A comprehensive survey and curated list of frameworks, hardware acceleration techniques, and applications for running LLMs on mobile and edge devices.
MLC-LLM: A machine learning compiler and high-performance deployment engine for large language models, supporting cross-platform accelerated execution. 
Awesome Lists & Curated Resources
Awesome Edge AI (wangxb96/Awesome-EdgeAI): A survey on data, model, and system optimizations for edge AI.
Awesome Cloud Edge AI (swagshaw/Awesome-Cloud-Edge-AI): A list of resources, including projects and papers, related to AI in cloud-edge synergy.
Awesome Edge Machine Learning (Bisonai/awesome-edge-machine-learning): A curated list of research papers, inference engines, and other resources. 
Collaborative Inference & Orchestration
KubeEdge: A Kubernetes-native edge computing framework that extends cloud computing capabilities to edge nodes, facilitating application deployment and management.
Cloud-edge collaborative inference for LLM (on KubeEdge-Ianvs): A specific GitHub issue/project aiming to build a framework for collaborative LLM inference between cloud and edge environments.
Collaborative Inference Work Summary (Tjyy-1223/Collaborative-Inference-Work-Summary): A repository focusing on DNN partitioning and task offloading strategies for cloud-edge systems. 
```

```
Frameworks and Toolkits
KubeEdge Sedna (kubeedge/sedna): An AI toolkit over KubeEdge that provides frameworks for edge-cloud synergy, including "joint inference" where difficult tasks are offloaded to the cloud.
ScorcaF/Edge-Cloud-Collaborative-Inference: A specific implementation of a collaborative inference framework featuring a fast inference pipeline on the edge and a slow/more complex one on the cloud, using a "success checker" to decide when to offload.
IoTDATALab/RTCoInfer: The implementation for the paper "RTCoInfer: Real-time Edge-Cloud Collaborative CNN Inference for Stream Analytics on Ubiquitous Images".
Tjyy-1223/Neurosurgeon: A repository implementing DNN partitioning for collaborative intelligence between the cloud and mobile edge, based on the "Neurosurgeon" concept. 
Awesome Lists and Research
swagshaw/Awesome-Cloud-Edge-AI: A curated list of systems for cloud and edge AI, including resources related to collaborative inference.
qijianpeng/awesome-edge-computing: A general list of edge computing resources.
ipc-lab/collaborative-inference-oac: Source code related to the research paper on "Private Collaborative Edge Inference via Over-the-Air Computation", focusing on privacy and bandwidth efficiency. 
Specific Applications
bytedance/Hybrid-SD: A novel framework for edge-cloud collaborative inference of Stable Diffusion models. 
These repositories offer both general frameworks and specific research implementations for various aspects of cloud-edge collaborative inference.
```

+ Aggregate results

```
  async def _execute_distributed_inference(self, request: InferenceRequest,
                                           execution_plan: Dict[str, Any]) -> InferenceResult:
        """Execute distributed inference across multiple edge devices."""
        
        device_ids = execution_plan["target_devices"]
        strategy = execution_plan.get("partitioning_strategy", ModelPartition.ENSEMBLE)
        
        # Create model partitions
        target_devices = [self.edge_devices[device_id] for device_id in device_ids]
        model_info = self.model_partitioner.analyze_model_structure(request.model_name)
        partitions = self.model_partitioner.create_partitions(model_info, target_devices, strategy)
        
        # Execute partitions in parallel
        partition_tasks = []
        for partition_id, partition_info in partitions["partitions"].items():
            task = self._execute_partition(request, partition_id, partition_info)
            partition_tasks.append(task)
        
        # Wait for all partitions to complete
        partition_results = await asyncio.gather(*partition_tasks, return_exceptions=True)
        
        # Aggregate results
        successful_results = [r for r in partition_results if isinstance(r, dict)]
        
        if not successful_results:
            raise Exception("All partition executions failed")
        
        # Simple aggregation (in real implementation, this would be more sophisticated)
        if partitions["strategy"] == "ensemble":
            aggregated_confidence = np.mean([r["confidence"] for r in successful_results])
            aggregated_result = {"prediction": successful_results[0]["prediction"], 
                               "confidence": aggregated_confidence}
        else:
            # For layer-wise, combine outputs sequentially
            aggregated_result = successful_results[-1]  # Simplified
        
        total_execution_time = max(r["execution_time"] for r in successful_results)
        
        return InferenceResult(
            request_id=request.request_id,
            result=aggregated_result,
            execution_time_ms=total_execution_time,
            device_id="+".join(device_ids),
            model_version="v1.0",
            confidence_score=aggregated_result.get("confidence", 0.0),
            processing_mode="distributed",
            metadata={
                "partitions_used": len(successful_results),
                "strategy": partitions["strategy"],
                "devices": device_ids
            }
        )
```

# ATEv2

```
docker exec -it  computenode2 bash
```

+  server   

```
pip3 install  zmq
```

```
root@5c943cb9786f:/pytorch/Vate-V2/edge-server# python3 main.py --no-jetson 
Got: ipc=False, jetson=False
```


+ edge   
```
 pip3 install opencv-python
```

```
mkdir -p /tmp/edge-device
python3 main.py "../detection-models/datasets/VisDrone/VisDrone2019-VID-test-dev/sequences/*" "../detection-models/datasets/VisDrone/annotations-VisDrone2019-VID-test-dev.json" --detection-rate 10
```