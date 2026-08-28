import os
import json
import datasets

_CITATION = """
@inproceedings{krishna2017dense,
    title={Dense-Captioning Events in Videos},
    author={Krishna, Ranjay and Hata, Kenji and Ren, Frederic and Fei-Fei, Li and Niebles, Juan Carlos},
    booktitle={International Conference on Computer Vision (ICCV)},
    year={2017}
}
"""

_DESCRIPTION = """\
The ActivityNet Captions dataset connects videos to a series of temporally annotated sentence descriptions.
Each sentence covers an unique segment of the video, describing multiple events that occur. These events
may occur over very long or short periods of time and are not limited in any capacity, allowing them to 
co-occur. On average, each of the 20k videos contains 3.65 temporally localized sentences, resulting in
a total of 100k sentences. We find that the number of sentences per video follows a relatively normal
distribution. Furthermore, as the video duration increases, the number of sentences also increases. 
Each sentence has an average length of 13.48 words, which is also normally distributed. You can find more
details of the dataset under the ActivityNet Captions Dataset section, and under supplementary materials 
in the paper.
"""

_URL_BASE = "https://cs.stanford.edu/people/ranjaykrishna/densevid/"


class ActivityNetConfig(datasets.BuilderConfig):
    """BuilderConfig for ActivityNet Captions."""

    def __init__(self, **kwargs):
        super(ActivityNetConfig, self).__init__(
            version=datasets.Version("2.1.0", ""), **kwargs)


class ActivityNet(datasets.GeneratorBasedBuilder):

    DEFAULT_CONFIG_NAME = "all"
    BUILDER_CONFIGS = [
        ActivityNetConfig(
            name="all", description="All the ActivityNet Captions dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "video_id": datasets.Value("string"),
                    "video_path": datasets.Value("string"),
                    "duration": datasets.Value("float32"),
                    "captions_starts": datasets.features.Sequence(datasets.Value("float32")),
                    "captions_ends": datasets.features.Sequence(datasets.Value("float32")),
                    "en_captions": datasets.features.Sequence(datasets.Value("string"))
                }
            ),
            supervised_keys=None,
            homepage=_URL_BASE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        archive_path = dl_manager.download_and_extract(
            _URL_BASE + "captions.zip")

        train_splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "infos_file": os.path.join(archive_path, "train.json")
                },
            )
        ]
        dev_splits = [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "infos_file": os.path.join(archive_path, "val_1.json")
                },
            )
        ]
        test_splits = [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "infos_file": os.path.join(archive_path, "val_2.json")
                },
            )
        ]
        return train_splits + dev_splits + test_splits

    def _generate_examples(self, infos_file):
        """This function returns the examples."""

        with open(infos_file, encoding="utf-8") as json_file:
            infos = json.load(json_file)
            for idx, id in enumerate(infos):
                path = "https://www.youtube.com/watch?v=" + id[2:]
                starts = [float(timestamp[0])
                          for timestamp in infos[id]["timestamps"]]
                ends = [float(timestamp[1])
                        for timestamp in infos[id]["timestamps"]]
                captions = [str(caption) for caption in infos[id]["sentences"]]
                yield idx, {
                    "video_id": id,
                    "video_path": path,
                    "duration": float(infos[id]["duration"]),
                    "captions_starts": starts,
                    "captions_ends": ends,
                    "en_captions": captions,
                }
                