from typing import Optional, List, Tuple
from PIL import Image
import argparse
import clip
from vqgan_utils import (
    load_vqgan_model,
    MakeCutouts,
    parse_prompt,
    resize_image,
    Prompt,
    synth,
    checkin,
    TVLoss,
)
import torch
from torchvision.transforms import functional as TF
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torchvision import transforms
import cv2
import numpy as np
import kornia.augmentation as K
from torch_optimizer import DiffGrad, AdamP, RAdam


# Set the optimiser
def get_opt(opt_name, z, opt_lr):
    """
    List of optimizers 
    Adadelta: Implements Adadelta algorithm.
    Adagrad: Implements Adagrad algorithm.
    Adam: Implements Adam algorithm.
    AdamW: Implements AdamW algorithm.
    Adamax: Implements Adamax algorithm (a variant of Adam based on infinity norm).
    ASGD: Implements Averaged Stochastic Gradient Descent.
    NAdam: Implements NAdam algorithm.
    RAdam: Implements RAdam algorithm.
    RMSprop: Implements RMSprop algorithm.
    Rprop: Implements the resilient backpropagation algorithm.
    SGD: Implements stochastic gradient descent (optionally with momentum)."""
    
    if opt_name == "Adam":
        opt = optim.Adam([z], lr=opt_lr)
    elif opt_name == "AdamW":
        opt = optim.AdamW([z], lr=opt_lr)
    elif opt_name == "Adagrad":
        opt = optim.Adagrad([z], lr=opt_lr)
    elif opt_name == "Adamax":
        opt = optim.Adamax([z], lr=opt_lr)   
    elif opt_name == "AdamP":
        opt = AdamP([z], lr=opt_lr)     
    elif opt_name == "Adadelta":
        opt = optim.Adadelta([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9)
    elif opt_name == "ASGD":
        opt = optim.ASGD([z], lr=opt_lr)
    elif opt_name == "DiffGrad":
        opt = DiffGrad([z], lr=opt_lr, eps=1e-9, weight_decay=1e-9)
    elif opt_name == "NAdam":
        opt = optim.NAdam([z], lr=opt_lr)
    elif opt_name == "RAdam":
        opt = RAdam([z], lr=opt_lr)
    elif opt_name == "RMSprop":
        opt = optim.RMSprop([z], lr=opt_lr)
    elif opt_name == "Rprop":
        opt = optim.Rprop([z], lr=opt_lr)
    elif opt_name == "SGD":
        opt = optim.SGD([z], lr=opt_lr)  
    
    else:
        print(f"Unknown optimiser: {opt_name} | Are choices broken?")
        opt = optim.Adam([z], lr=opt_lr)
    return opt

class Run:
    """
    Subclass this to house your own implementation of CLIP-based image generation 
    models within the UI
    """

    def __init__(self):
        """
        Set up the run's config here
        """
        pass

    def load_model(self):
        """
        Load models here. Separated this from __init__ to allow loading model state
        from a previous run
        """
        pass

    def model_init(self):
        """
        Continue run setup, for items that require the models to be in=place. 
        Call once after load_model
        """
        pass

    def iterate(self):
        """
        Place iteration logic here. Outputs results for human consumption at
        every step. 
        """
        pass


class VQGANCLIPRun(Run):
    def __init__(
        # Inputs
        self,
        text_input: str = "the first day of the waters",
        vqgan_ckpt: str = "vqgan_imagenet_f16_16384",
        clip_model: str = "ViT-B/32",
        num_steps: int = 300,
        image_x: int = 300,
        image_y: int = 300,
        init_image: Optional[Image.Image] = None,
        image_prompts: List[Image.Image] = [],
        continue_prev_run: bool = False,
        seed: Optional[int] = None,
        cutn: int = 32,
        cut_pow: float = 1.0,
        step_size: float = 0.05,
        opt_name: str = "Adam",
        mse_weight=0.5,
        mse_weight_decay=0.1,
        mse_weight_decay_steps=50,
        tv_loss_weight=0.000085,
        use_cutout_augmentations: bool = True,
        # use_augs: bool = True,
        # noise_fac: float = 0.1,
        # use_noise: Optional[float] = None,
        # mse_withzeros=True,
        ## **kwargs,  # Use this to receive Streamlit objects ## Call from main UI
        use_scrolling_zooming: bool = False,
        translation_x: int = 0,
        translation_y: int = 0,
        rotation_angle: float = 0,
        zoom_factor: float = 1,
        transform_interval: int = 10,
        device: Optional[torch.device] = "cpu",
    ) -> None:
        super().__init__()
        self.text_input = text_input
        self.vqgan_ckpt = vqgan_ckpt
        self.clip_model = clip_model
        self.num_steps = num_steps
        self.image_x = image_x
        self.image_y = image_y
        self.init_image = init_image
        self.image_prompts = image_prompts
        self.continue_prev_run = continue_prev_run
        self.seed = seed
        self.cutn = cutn
        self.cut_pow = cut_pow
        self.step_size = step_size
        self.opt_name = opt_name
        self.device = device

        # Setup ------------------------------------------------------------------------------
        # Split text by "|" symbol
        texts = [phrase.strip() for phrase in text_input.split("|")]
        if texts == [""]:
            texts = []

        # Leaving most of this untouched
        self.args = argparse.Namespace(
            prompts=texts,
            image_prompts=image_prompts,
            noise_prompt_seeds=[],
            noise_prompt_weights=[],
            size=[int(image_x), int(image_y)],
            init_image=init_image,
            init_weight=mse_weight,
            # clip.available_models()
            # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px']
            # Visual Transformer seems to be the smallest
            clip_model=clip_model,
            vqgan_config=f"assets/{vqgan_ckpt}.yaml",
            vqgan_checkpoint=f"assets/{vqgan_ckpt}.ckpt",
            cutn=cutn,
            cut_pow=cut_pow,
            step_size=step_size,
            opt_name=opt_name,
            display_freq=50,
            seed=seed,
            device=device,
        )

        if device is None or device == "cpu":
            #self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            device = torch.device("cpu")            
        else:
            #self.device = device
            self.device = torch.device(f"{device}" if torch.cuda.is_available() else "cpu")

        print("Using device:", device)

        self.iterate_counter = 0
        # self.use_augs = use_augs
        # self.noise_fac = noise_fac
        # self.use_noise = use_noise
        # self.mse_withzeros = mse_withzeros
        self.init_mse_weight = mse_weight
        self.mse_weight = mse_weight
        self.mse_weight_decay = mse_weight_decay
        self.mse_weight_decay_steps = mse_weight_decay_steps

        self.use_cutout_augmentations = use_cutout_augmentations

        # For TV loss
        self.tv_loss_weight = tv_loss_weight

        self.use_scrolling_zooming = use_scrolling_zooming
        self.translation_x = translation_x
        self.translation_y = translation_y
        self.rotation_angle = rotation_angle
        self.zoom_factor = zoom_factor
        self.transform_interval = transform_interval
        
        

    def load_model(
        self, prev_model: nn.Module = None, prev_perceptor: nn.Module = None
    ) -> Optional[Tuple[nn.Module, nn.Module]]:
        if self.continue_prev_run is True:
            self.model = prev_model
            self.perceptor = prev_perceptor
            return None

        else:
            self.model = load_vqgan_model(
                self.args.vqgan_config, self.args.vqgan_checkpoint
            ).to(self.device)

            self.perceptor = (
                clip.load(self.args.clip_model, jit=False)[0]
                .eval()
                .requires_grad_(False)
                .to(self.device)
            )

            return self.model, self.perceptor

    def model_init(self, init_image: Image.Image = None) -> None:
        cut_size = self.perceptor.visual.input_resolution
        e_dim = self.model.quantize.e_dim
        f = 2 ** (self.model.decoder.num_resolutions - 1)

        if self.use_cutout_augmentations:
            noise_fac = 0.1
            augs = nn.Sequential(
                K.RandomHorizontalFlip(p=0.5),
                K.RandomSharpness(0.3, p=0.4),
                K.RandomAffine(degrees=30, translate=0.1, p=0.8, padding_mode="border"),
                K.RandomPerspective(0.2, p=0.4),
                K.ColorJitter(hue=0.01, saturation=0.01, p=0.7),
            )
        else:
            noise_fac = None
            augs = None

        self.make_cutouts = MakeCutouts(
            cut_size,
            self.args.cutn,
            cut_pow=self.args.cut_pow,
            noise_fac=noise_fac,
            augs=augs,
        )

        n_toks = self.model.quantize.n_e
        toksX, toksY = self.args.size[0] // f, self.args.size[1] // f
        sideX, sideY = toksX * f, toksY * f
        self.z_min = self.model.quantize.embedding.weight.min(dim=0).values[
            None, :, None, None
        ]
        self.z_max = self.model.quantize.embedding.weight.max(dim=0).values[
            None, :, None, None
        ]


        if self.seed is not None:
            torch.manual_seed(self.seed)
        else:
            self.seed = torch.seed()  # Trigger a seed, retrieve the utilized seed

        # Initialization order: continue_prev_im, init_image, then only random init
        if init_image is not None:
            init_image = init_image.resize((sideX, sideY), Image.LANCZOS)
            self.z, *_ = self.model.encode(
                TF.to_tensor(init_image).to(self.device).unsqueeze(0) * 2 - 1
            )
        elif self.args.init_image:
            pil_image = self.args.init_image
            pil_image = pil_image.resize((sideX, sideY), Image.LANCZOS)
            self.z, *_ = self.model.encode(
                TF.to_tensor(pil_image).to(self.device).unsqueeze(0) * 2 - 1
            )
        else:
            one_hot = F.one_hot(
                torch.randint(n_toks, [toksY * toksX], device=self.device), n_toks
            ).float()
            self.z = one_hot @ self.model.quantize.embedding.weight
            self.z = self.z.view([-1, toksY, toksX, e_dim]).permute(0, 3, 1, 2)
        self.z_orig = self.z.clone()
        self.z.requires_grad_(True)
        #self.opt = optim.Adam([self.z], lr=self.args.step_size)
        self.opt = get_opt(self.opt_name, self.z, self.args.step_size)

        self.normalize = transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711],
        )

        self.pMs = []

        for prompt in self.args.prompts:
            txt, weight, stop = parse_prompt(prompt)
            embed = self.perceptor.encode_text(
                clip.tokenize(txt).to(self.device)
            ).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for uploaded_image in self.args.image_prompts:
            # path, weight, stop = parse_prompt(prompt)
            # img = resize_image(Image.open(fetch(path)).convert("RGB"), (sideX, sideY))
            img = resize_image(uploaded_image.convert("RGB"), (sideX, sideY))
            batch = self.make_cutouts(TF.to_tensor(img).unsqueeze(0).to(self.device))
            embed = self.perceptor.encode_image(self.normalize(batch)).float()
            self.pMs.append(Prompt(embed, weight, stop).to(self.device))

        for seed, weight in zip(
            self.args.noise_prompt_seeds, self.args.noise_prompt_weights
        ):
            gen = torch.Generator().manual_seed(seed)
            embed = torch.empty([1, self.perceptor.visual.output_dim]).normal_(
                generator=gen
            )
            self.pMs.append(Prompt(embed, weight).to(self.device))

    def _ascend_txt(self) -> List:
        out = synth(self.model, self.z)
        iii = self.perceptor.encode_image(
            self.normalize(self.make_cutouts(out))
        ).float()

        result = {}

        if self.args.init_weight:
            result["mse_loss"] = F.mse_loss(self.z, self.z_orig) * self.mse_weight / 2

            # MSE regularization scheduler
            with torch.no_grad():
                # if not the first step
                # and is time for step change
                # and both weight decay steps and magnitude are nonzero
                # and MSE isn't zero already
                if (
                    self.iterate_counter > 0
                    and self.iterate_counter % self.mse_weight_decay_steps == 0
                    and self.mse_weight_decay != 0
                    and self.mse_weight_decay_steps != 0
                    and self.mse_weight != 0
                ):
                    self.mse_weight = self.mse_weight - self.mse_weight_decay

                    # Don't allow changing sign
                    # Basically, caps MSE at zero if decreasing from positive
                    # But, also prevents MSE from becoming positive if -MSE intended
                    if self.init_mse_weight > 0:
                        self.mse_weight = max(self.mse_weight, 0)
                    else:
                        self.mse_weight = min(self.mse_weight, 0)

                    print(f"updated mse weight: {self.mse_weight}")

        tv_loss_fn = TVLoss()
        result["tv_loss"] = tv_loss_fn(self.z) * self.tv_loss_weight

        for count, prompt in enumerate(self.pMs):
            result[f"prompt_loss_{count}"] = prompt(iii)

        return result
    
    def iterate(self) -> Tuple[List[float], Image.Image]:
        if not self.use_scrolling_zooming:
            # Forward prop
            self.opt.zero_grad()
            #for param in self.model.parameters():
                #param.grad = None
                
            losses = self._ascend_txt()

            # Grab an image
            im: Image.Image = checkin(self.model, self.z)

            # Backprop
            loss = sum([j for i, j in losses.items()])
            loss.backward()
            self.opt.step()
            with torch.no_grad():
                self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))

            # Advance iteration counter
            self.iterate_counter += 1

            print(
                f"Step {self.iterate_counter} losses: {[(i, j.item()) for i, j in losses.items()]}"
            )

            # Output stuff useful for humans
            return [(i, j.item()) for i, j in losses.items()], im

        else:
            # Grab current image
            im_before_transform: Image.Image = checkin(self.model, self.z)

            # Convert for use in OpenCV
            imarr = np.array(im_before_transform)
            imarr = cv2.cvtColor(imarr, cv2.COLOR_RGB2BGR)

            translation = np.float32(
                [[1, 0, self.translation_x], [0, 1, self.translation_y]]
            )

            imcenter = (imarr.shape[1] // 2, imarr.shape[0] // 2)
            rotation = cv2.getRotationMatrix2D(
                imcenter, angle=self.rotation_angle, scale=self.zoom_factor
            )

            trans_mat = np.vstack([translation, [0, 0, 1]])
            rot_mat = np.vstack([rotation, [0, 0, 1]])
            transformation_matrix = np.matmul(rot_mat, trans_mat)

            outarr = cv2.warpPerspective(
                imarr,
                transformation_matrix,
                (imarr.shape[1], imarr.shape[0]),
                borderMode=cv2.BORDER_WRAP,
            )

            transformed_im = Image.fromarray(cv2.cvtColor(outarr, cv2.COLOR_BGR2RGB))

            # Encode as z, reinit
            self.z, *_ = self.model.encode(
                TF.to_tensor(transformed_im).to(self.device).unsqueeze(0) * 2 - 1
            )
            self.z.requires_grad_(True)
            
            #self.opt = optim.Adam([self.z], lr=self.args.step_size)
            self.opt = get_opt(self.opt_name, self.z, self.args.step_size)

            for _ in range(self.transform_interval):
                # Forward prop
                self.opt.zero_grad()
                #for param in self.model.parameters():
                    #param.grad = None
                    
                losses = self._ascend_txt()

                # Grab an image
                im: Image.Image = checkin(self.model, self.z)

                # Backprop
                loss = sum([j for i, j in losses.items()])
                loss.backward()
                self.opt.step()
                with torch.no_grad():
                    self.z.copy_(self.z.maximum(self.z_min).minimum(self.z_max))
                    
            # Advance iteration counter
            self.iterate_counter += 1
            
            for param_group in self.opt.param_groups:
                #print (param_group)    
                print (f"Learning Rate: {param_group['lr']}")

            print(
                f"Step {self.iterate_counter} losses: {[(i, j.item()) for i, j in losses.items()]}"
            )

            # Output stuff useful for humans
            return [(i, j.item()) for i, j in losses.items()], im
