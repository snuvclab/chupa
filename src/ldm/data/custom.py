import os
from pathlib import Path
import random
import numpy as np
import PIL
import cv2
from PIL import Image, ImageOps
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
from functools import partial

from taming.data.base import ImagePaths, NumpyPaths, ConcatDatasetWithIndex
from modules.image_degradation import degradation_fn_bsr_light


class CustomBase(Dataset):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.data = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        example = self.data[i]
        return example

class CustomTrain(CustomBase):
    def __init__(self, size, training_images_list_file, use_data_root=False, nsml=False):
        super().__init__()
        if nsml:
            from nsml import DATASET_PATH
            dataset = Path(training_images_list_file).stem.split("_")[0]
            paths = sorted(list(Path(f"{DATASET_PATH}/train").joinpath(dataset).glob("*.png")))
            paths = [str(x) for x in paths]
        else:
            with open(training_images_list_file, "r") as f:
                paths = f.read().splitlines()
            if use_data_root:
                data_root = Path(training_images_list_file)
                data_root = data_root.parent.joinpath(data_root.stem.split("_")[0])
                paths = [str(data_root.joinpath(p)) for p in paths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)

class CustomTest(CustomBase):
    def __init__(self, size, test_images_list_file, use_data_root=False, nsml=False):
        super().__init__()
        if nsml:
            from nsml import DATASET_PATH
            dataset = Path(test_images_list_file).stem.split("_")[0]
            paths = sorted(list(Path(DATASET_PATH).joinpath(f"train/{dataset}").glob("*.png")))
            paths = [str(x) for x in paths]
        else:
            with open(test_images_list_file, "r") as f:
                paths = f.read().splitlines()
            if use_data_root:
                data_root = Path(test_images_list_file)
                data_root = data_root.parent.joinpath(data_root.stem.split("_")[0])
                paths = [str(data_root.joinpath(p)) for p in paths]
        self.data = ImagePaths(paths=paths, size=size, random_crop=False)


class LSUNBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5, key=None
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        self.labels = {
            "relative_file_path_": [l for l in self.image_paths],
            "file_path_": [os.path.join(self.data_root, l)
                           for l in self.image_paths],
        }

        self.size = size
        self.interpolation = Image.BICUBIC
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)
        self.key = "image" if key == None else key

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w, = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2, (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example[self.key] = (image / 127.5 - 1.0).astype(np.float32)
        return example

class ShapeNet(Dataset):
    def __init__(self, root_path=None, split="train", 
                class_label="02958343",
                keys=["normal", "T_normal"],
                num_views=36, size=512, img_type="RGB", 
                use_uc_prob=0.2,
                dual=True):
            
        self.data_root = Path(root_path) / class_label / split
        self.split = split
        self.keys = keys
        self.dual = dual
        self.num_views = num_views
        self.yaw_interval = int(360 / num_views)
        self.size = size
        self.img_type = img_type
        self.use_uc_prob = use_uc_prob

        self.subjects = list(self.data_root.iterdir())

    def __len__(self):
        return len(self.subjects) * self.num_views

    def __getitem__(self, idx):
        example = {}
        subject = self.subjects[idx // self.num_views]
        view_angle = (idx % self.num_views) * self.yaw_interval
        uncond = True if (np.random.rand() < self.use_uc_prob) else False 
        for key in self.keys:
            img_name = subject / (key + "_F") / f"{view_angle:03d}.png"
            image = Image.open(img_name).convert(self.img_type)
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=Image.BICUBIC)
            image = np.array(image).astype(np.uint8)

            if self.split == "train" and key.startswith("T_") and uncond:
                # randomly zero out the condition for cfg
                example[key + "_F"] = -np.ones_like(image).astype(np.float32)
            else:
                example[key + "_F"] = (image / 127.5 - 1.0).astype(np.float32)

            # back side normal map
            if not key.startswith("T_") and self.dual:  # not for template
                back_img_name = subject / (key + "_F") / f"{(view_angle + 180) % 360:03d}.png"
                back_image = Image.open(back_img_name)
                if self.size is not None:
                    back_image = back_image.resize((self.size, self.size), resample=Image.BICUBIC)
                back_image_fliplr = np.fliplr(np.array(back_image))
                # flip horizontally
                mask = (back_image_fliplr[:,:,3] > 0)
                back_image_fliplr[mask, 0] = -back_image_fliplr[mask, 0]
                example[key + "_B"] = (back_image_fliplr[:, :, :3] / 127.5 - 1.0).astype(np.float32)
                
        return example

class NormalData(Dataset):
    def __init__(self, root_path=None, split="train", 
                class_label="",
                keys=["normal_F", "T_normal_F"],
                num_views=36, size=512, img_type="RGBA", 
                use_uc_prob=0.2,
                dual=True):
            
        self.data_root = Path(root_path) / class_label / 'render' / split
        self.split = split
        self.keys = keys
        self.dual = dual
        self.num_views = num_views
        self.yaw_interval = int(360 / num_views)
        self.size = size
        self.img_type = img_type
        self.num_ch = 3 if img_type == 'RGB' else 4
        self.use_uc_prob = use_uc_prob

        self.subjects = list(self.data_root.iterdir())

    def __len__(self):
        return len(self.subjects) * self.num_views

    def __getitem__(self, idx):
        example = {}
        subject = self.subjects[idx // self.num_views]
        view_angle = (idx % self.num_views) * self.yaw_interval
        uncond = True if (np.random.rand() < self.use_uc_prob) else False 
        for key in self.keys:
            img_name = subject / (key + "_F") / f"{view_angle:03d}.png"
            image = Image.open(img_name).convert(self.img_type)
            if self.size is not None:
                image = image.resize((self.size, self.size), resample=Image.BICUBIC)
            image = np.array(image).astype(np.uint8)

            key = key[:-5] if '_face' in key else key
            if self.split == "train" and key.startswith("T_") and uncond:
                # randomly zero out the condition for cfg
                example[key + "_F"] = -np.ones_like(image).astype(np.float32)
            else:
                example[key + "_F"] = (image / 127.5 - 1.0).astype(np.float32)

            # back side normal map
            if not key.startswith("T_") and self.dual:  # not for template (SMPL-X)
                back_img_name = subject / (key + "_F") / f"{(view_angle + 180) % 360:03d}.png"
                back_image = Image.open(back_img_name)
                if self.size is not None:
                    back_image = back_image.resize((self.size, self.size), resample=Image.BICUBIC)
                back_image_fliplr = np.fliplr(np.array(back_image))
                # flip horizontally
                mask = (back_image_fliplr[:,:,3] > 0)
                back_image_fliplr[mask, 0] = -back_image_fliplr[mask, 0]
                example[key + "_B"] = (back_image_fliplr[:, :, :self.num_ch] / 127.5 - 1.0).astype(np.float32)
                
        return example
        
class TigerbroTrain(LSUNBase):
    def __init__(self, nsml=False, **kwargs):
        if nsml:
            from nsml import DATASET_PATH
            txt_file = os.path.join(DATASET_PATH, "train/tigerbro_256_wild-train.txt")
            data_root = os.path.join(DATASET_PATH, "train/tigerbro_256_wild")
        else:
            txt_file = "/datasets/RD/nsml-RD/cartoon_data/tigerbro/tigerbro_256_wild-train.txt"
            data_root = "/datasets/RD/nsml-RD/cartoon_data/tigerbro/tigerbro_256_wild"
        super().__init__(txt_file=txt_file, data_root=data_root, **kwargs)

class TigerbroValidation(LSUNBase):
    def __init__(self, flip_p=0., nsml=False, **kwargs):
        if nsml:
            from nsml import DATASET_PATH
            txt_file = os.path.join(DATASET_PATH, "train/tigerbro_256_wild-valid.txt")
            data_root = os.path.join(DATASET_PATH, "train/tigerbro_256_wild")
        else:
            txt_file = "/datasets/RD/nsml-RD/cartoon_data/tigerbro/tigerbro_256_wild-valid.txt"
            data_root = "/datasets/RD/nsml-RD/cartoon_data/tigerbro/tigerbro_256_wild"
        super().__init__(txt_file=txt_file, data_root=data_root, flip_p=flip_p, **kwargs)

class SketchImgData(Dataset):
    def __init__(self, type="train", mode="RGB", root_path=None, nsml=False, 
        size=256, downscale_f=1, use_canny=False, use_degrade=True,
        min_crop=0.5, max_crop=1.0, random_crop=True):
        if root_path == None:
            if nsml:
                from nsml import DATASET_PATH
                root_path = Path(DATASET_PATH).joinpath("train")
            else:
                root_path = Path("/datasets/RD/pixivSCpair")
        else:
            root_path = Path(root_path)
            
        self.img_path = root_path.joinpath("trainA" if type == "train" else "testA")
        self.sketch_path = root_path.joinpath("trainB" if type == "train" else "testB")
        self.img_files = sorted(list(self.img_path.glob("*.jpg"))+list(self.img_path.glob("*.png")))
        self.sketch_files = sorted(list(self.sketch_path.glob("*.jpg"))+list(self.sketch_path.glob("*.png")))
        assert len(self.img_files) == len(self.sketch_files)
        
        self.mode = mode
        self.size = size
        self.down_size = size // downscale_f
        self.min_crop_f = min_crop
        self.max_crop_f = max_crop
        self.center_crop = not random_crop
        self.use_canny = use_canny
        self.use_degrade = use_degrade

        self.image_rescaler = A.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, i):
        img_name = self.img_files[i]
        sketch_name = self.sketch_path.joinpath(img_name.name)
        
        image = Image.open(img_name).convert(self.mode)
        sketch = Image.open(sketch_name).convert(self.mode)
        if sketch.size != image.size:
            sketch = sketch.resize(image.size, resample=Image.BICUBIC)

        image = np.array(image).astype(np.uint8)
        image = np.expand_dims(image, axis=-1) if self.mode == "L" else image
        sketch = np.array(sketch).astype(np.uint8)
        sketch = np.expand_dims(sketch, axis=-1) if self.mode == "L" else sketch

        min_side_len = min(image.shape[:2])
        crop_side_len = int(min_side_len * np.random.uniform(
            self.min_crop_f, self.max_crop_f, size=None
        ))
        kwargs = {"height" : crop_side_len, "width" : crop_side_len}
        A_transf = A.Compose([
            A.CenterCrop(**kwargs) if self.center_crop else A.RandomCrop(**kwargs), 
            self.image_rescaler
        ], additional_targets={"sketch" : "image"})
        transformed = A_transf(image=image, sketch=sketch)

        if self.mode != "L" and self.use_degrade:
            image = self.degradation_process(image=transformed["image"])["image"]
        else:
            image = transformed["image"]
        sketch = transformed["sketch"]

        if self.use_canny:
            image = np.expand_dims(cv2.Canny(image, 200, 250), 2).repeat(3, axis=2)

        example = {"relpath" : str(img_name)}
        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        example["sketch"] = (sketch/127.5 - 1.0).astype(np.float32)

        return example
        


