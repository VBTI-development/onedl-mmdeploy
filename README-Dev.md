This ReadMe is for developers who want to use this repo.

## Devcontainer

There is a devcontainer that allows for development of both the C++ and the python code.
The post-install-script will install the code bases (e.g. mmdet) you are developing for.
Currently we only support mmdet and mmpretrain.

If another code base is added the post-install script should install it.
Furthermore the buildcmd.sh should be updated to build mmdeploy for that code base too,
as well as the cmake command in the Dockerfile used for the devcontainer.

## Testing

### CPP tests

There is a cpp test executable included.

The tests that are run are defined in tests/csrc/
If you add a cpp file in one of the subfolders with new tests, they should be picked up automatically.

### Python tests

The python test extension in VSCode should automatically pick up the python tests in the tests folder.
Run them with pytest.

Some things to keep in mind.

1. Some tests are skipped automatically if the backend (e.g. tensorrt) is not installed.
2. Some tests with tensorrt backend don't clean up nicely failing the next tests (e.g. test_grid_sample).
   But when they are run separately they are succeeding.

### End to end tests

To run the full pipeline tests of certain models, run any of the `scripts/testing/test_full_pipeline...` scripts.
Run them with one of the following argument options:

1. `ort cpu` to run onnx conversion and test on cpu.
2. `trt cuda` to run TRT conversion and test on GPU (requires GPU to be available).
3. `ort gpu` to run onnx conversion and test on GPU (!does not always work because of dependency issues).

The scripts will:

1. Create a folder `work_dir_<model_name>` to store files.
2. Download the pretrained weights of the model and model definition files.
3. Convert the model to the requested back end.
4. Download a test dataset and store in the data folder (imagenet for pretrain and tiny_coco for mmdet).
5. Run the model using only the backed and print evaluation results
6. Run the model using the SDK (with python bindings) and print evaluation results.
   These results should be close to the results in the previous step.
7. Run the model using the SDK with a fixed image size and print the FPS results.

To create a new model, simply one of the scripts and update:

1. `work_dir` to make a new work dir
2. `model_cfg` to match the config file that is downloaded with the `mim download` call
3. `checkpoint` to match the checkpoint file that is downloaded with the `mim download` call.
4. `sdk_cfg` to match object detection or instance segmentation sdk config.

! Note: rmt-det-ins has a special option that needs to be present in the deploy config files, hence the extra lines.

#### Visualize results

To visualize mmdetection outputs, add the following code to `mmdeploy/codebase/mmdet/deploy/object_detection_model.py`.
This will create a `tmp_output` folder and stores the predictions there.

```python
        from mmdet.visualization.local_visualizer import DetLocalVisualizer
        visualizer = DetLocalVisualizer(save_dir="tmp_output")
        visualizer.add_datasample(
            data_samples[0].img_path.split('/')[-1],
            inputs.contiguous().detach().cpu().numpy()[...,::-1],
            data_samples[0],
            draw_gt=False,
            draw_pred=True,
            out_file="tmp_output/"+data_samples[0].img_path.split('/')[-1],
        )
```
