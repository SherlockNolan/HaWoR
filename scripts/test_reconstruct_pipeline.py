from reconstruct import HaWoRPipeline


if __name__ == "__main__":
    reconstructor = HaWoRPipeline(
        checkpoint="./weights/hawor/checkpoints/hawor.ckpt",
        infiller_weight="./weights/hawor/checkpoints/infiller.pt",
    )
    result = reconstructor.reconstruct("example/video_0.mp4", output_dir="./output", rendering=True, vis_mode="cam")
