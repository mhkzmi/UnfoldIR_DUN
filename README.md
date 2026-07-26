<div align="center">

<h1>UnfoldIR: Rethinking Deep Unfolding Network in Illumination Degradation Image Restoration</h1>

[![arXiv](https://img.shields.io/badge/arXiv-2505.06683-b31b1b.svg)](https://arxiv.org/abs/2505.06683)&nbsp;
[![My Presentation](https://img.shields.io/badge/My_Presentation-Slides-blue)](./docs/UnfoldIR_Slides.pptx)&nbsp;
[![My Report](https://img.shields.io/badge/My_Report-PDF-yellow)](./docs/UnfoldIR_Summary.pdf)&nbsp;


**Implemented by:** Mohammad Kazemi &nbsp;|&nbsp; **Supervised by:** Dr. H.A. GhiassiRad  
**Institution:** K. N. Toosi University of Technology

*Based on the original paper by Chunming He, et al.*

</div>

<div dir="rtl" align="right">

# پروژه UnfoldIR CPU

این پروژه یک پیاده‌سازی آموزشی، قابل اجرا و CPU-only از ایده‌های مقاله فوق است. هدف، ساختن یک مدل trainable بر پایه ایده‌های مقاله برای بهبود تصویرهای کم‌نور، backlit، underwater، fundus و تصویرهای دچار تخریب روشنایی است.

این پروژه کپی رسمی مقاله یا وزن رسمی نیست. اگر وزن رسمی و دیتاست مقاله در اختیار نباشد، خروجی دقیقاً برابر مقاله نخواهد بود. مدل این پروژه با وزن اولیه هم inference انجام می‌دهد، اما کیفیت اصلی وقتی بهتر می‌شود که training با دیتاست مناسب اجرا شود.

## ایده‌های استفاده‌شده

- تجزیه Retinex با رابطه `I = R * L`
- initialization با `L0 = max RGB` و smoothing
- شبکه unfolding چندمرحله‌ای با `K=3`
- RAIC-Lite برای اصلاح illumination با کمک reflectance
- IGRE-Lite برای تقویت reflectance با راهنمایی illumination
- FVSS-Lite به جای VSS/Mamba اصلی برای اجرای سبک روی CPU
- Haar DWT/IDWT پیاده‌سازی‌شده با PyTorch و قابل آموزش
- refinement سبک شبیه RK2:
  `k1 = FVSS(R_hat, L)` و `k2 = FVSS(R_hat + k1, L)`
- gating برای کنترل شدت refinement
- ISIC loss روی stageهای پایانی برای پایداری بین مراحل

## محدودیت‌ها

- وزن رسمی مقاله داخل این پروژه وجود ندارد.
- نسخه اصلی مقاله با منابع محاسباتی سنگین‌تر آموزش داده شده است.
- این نسخه برای CPU و ویندوز سبک‌سازی شده است.
- VSS/Mamba اصلی با VSS-Lite جایگزین شده تا وابستگی خاص و اجرای سنگین نداشته باشد.
- خروجی بدون training محدود است و بیشتر نقش smoke/inference اولیه دارد.
- برای نزدیک شدن به خروجی مقاله باید دیتاست paired مناسب و آموزش کافی داشته باشید.

## نصب

در پوشه پروژه اجرا کنید:

```
pip install -r requirements.txt
```


## اجرای GUI

```
python app_gui.py
```

در GUI می‌توانید تصویر یا پوشه انتخاب کنید، checkpoint دلخواه بدهید، بهبود تصویر را اجرا کنید و پوشه `outputs` را باز کنید.

## اجرای inference

تصاویر را در این مسیر بگذارید:

```text
data/input
```

سپس اجرا کنید:

```
python infer.py --input data\input --output outputs
```

اگر `checkpoints/best_cpu.pth` موجود نباشد، برنامه با وزن اولیه اجرا می‌شود و پیام می‌دهد که برای کیفیت بهتر باید training انجام شود.

خروجی‌ها:

- `*_enhanced.png`
- `*_illumination.png`
- `*_reflectance.png`
- خروجی stageها در صورت فعال بودن debug/config

## آموزش

```
python train.py --checkpoint checkpoints\best_cpu.pth
```

checkpoint در این مسیر ذخیره می‌شود:

```text
checkpoints/best_cpu.pth
```

### حالت A: supervised paired

اگر در `data/train_low` و `data/train_high` فایل‌های هم‌نام وجود داشته باشند، برنامه از حالت paired استفاده می‌کند.

Lossها:

- L1
- SSIM loss
- gradient/edge loss
- color consistency
- illumination smoothness
- reflectance texture loss
- ISIC loss برای stageهای آخر

### حالت B: self-supervised / unpaired

اگر فقط `data/train_low` پر باشد، برنامه از lossهای Retinex reconstruction، exposure، illumination smoothness، color constancy، noise suppression، saturation penalty و ISIC استفاده می‌کند. کیفیت این حالت معمولاً از حالت paired پایین‌تر است.

### حالت C: synthetic training

اگر دیتاست آموزشی وجود نداشته باشد اما در `data/input` یا `data/test` تصویر باشد، برنامه از همان تصاویر یک نسخه pseudo-clean می‌سازد و low-light مصنوعی تولید می‌کند:

- gamma darkening
- random illumination mask
- color shift
- noise سبک
- backlit simulation

این حالت فقط برای یادگیری اولیه وزن‌هاست و ادعا نمی‌کند برابر دیتاست مقاله است.

## ساختار داده

```text
data/input       تصاویر inference
data/train_low   تصاویر کم‌نور آموزش
data/train_high  تصاویر مرجع هم‌نام برای paired training
data/test        تصاویر تست یا منبع synthetic training
outputs          خروجی‌های inference
checkpoints      checkpointهای آموزش
```

## پیشنهاد برای کیفیت بهتر

- اگر ممکن است دیتاست paired مثل LOL-v1 یا LOL-v2 را اضافه کنید.
- برای CPU مقدار `image_size=256` مناسب‌تر است.
- حداقل 20 تا 100 epoch آموزش بدهید؛ زمان آن به CPU سیستم بستگی دارد.
- اگر تصویر بزرگ است، inference پیش‌فرض از tile mode با overlap استفاده می‌کند تا ابعاد حفظ شود.

## تست سریع

بعد از نصب:

```
python infer.py --self-test
```

این تست یک tensor کم‌نور کوچک داخل حافظه می‌سازد و forward مدل را بررسی می‌کند. تصویر جعلی در data ساخته نمی‌شود.


## ارجاع به مقاله اصلی (Citation)
</div>

```bibtex
@misc{he2025unfoldirrethinkingdeepunfolding,
      title={UnfoldIR: Rethinking Deep Unfolding Network in Illumination Degradation Image Restoration}, 
      author={Chunming He and Rihan Zhang and Fengyang Xiao and Chengyu Fang and Longxiang Tang and Yulun Zhang and Sina Farsiu},
      year={2025},
      eprint={2505.06683},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2505.06683}, 
}
```
