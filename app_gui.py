import os
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import Button, Entry, Frame, Label, StringVar, Tk, filedialog, messagebox

from PIL import Image, ImageTk


ROOT = Path(__file__).resolve().parent
OUTPUTS = ROOT / "outputs"


class App:
    def __init__(self, root):
        self.root = root
        self.root.title("UnfoldIR - Illumination Degradation Image Restoration")
        self.input_path = StringVar()
        self.checkpoint_path = StringVar(value=str(ROOT / "checkpoints" / "best_cpu.pth"))
        self.status = StringVar(value="Ready")
        self.before_img = None
        self.after_img = None
        self._build()

    def _build(self):
        top = Frame(self.root, padx=10, pady=10)
        top.pack(fill="x")
        Button(top, text="Select Image", command=self.choose_image).pack(side="left", padx=4)
        Button(top, text="Selet Directory", command=self.choose_folder).pack(side="left", padx=4)
        Entry(top, textvariable=self.input_path, width=64).pack(side="left", padx=4, fill="x", expand=True)

        mid = Frame(self.root, padx=10, pady=4)
        mid.pack(fill="x")
        Button(mid, text="Select Checkpoint", command=self.choose_checkpoint).pack(side="left", padx=4)
        Entry(mid, textvariable=self.checkpoint_path, width=64).pack(side="left", padx=4, fill="x", expand=True)
        Button(mid, text="GO", command=self.run).pack(side="left", padx=4)
        Button(mid, text="Open Outputs", command=self.open_outputs).pack(side="left", padx=4)

        body = Frame(self.root, padx=10, pady=10)
        body.pack(fill="both", expand=True)
        self.before_label = Label(body, text="Before")
        self.before_label.pack(side="left", padx=8, fill="both", expand=True)
        self.after_label = Label(body, text="After")
        self.after_label.pack(side="left", padx=8, fill="both", expand=True)
        Label(self.root, textvariable=self.status, anchor="w", padx=10, pady=6).pack(fill="x")

    def choose_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff *.webp")])
        if path:
            self.input_path.set(path)
            self.show_before(path)

    def choose_folder(self):
        path = filedialog.askdirectory()
        if path:
            self.input_path.set(path)

    def choose_checkpoint(self):
        path = filedialog.askopenfilename(filetypes=[("PyTorch checkpoint", "*.pth *.pt"), ("All files", "*.*")])
        if path:
            self.checkpoint_path.set(path)

    def open_outputs(self):
        OUTPUTS.mkdir(exist_ok=True)
        os.startfile(OUTPUTS)

    def show_before(self, path):
        try:
            img = Image.open(path).convert("RGB")
            img.thumbnail((420, 420))
            self.before_img = ImageTk.PhotoImage(img)
            self.before_label.configure(image=self.before_img, text="")
        except Exception as exc:
            messagebox.showerror("Error", str(exc))

    def show_after(self, path):
        img = Image.open(path).convert("RGB")
        img.thumbnail((420, 420))
        self.after_img = ImageTk.PhotoImage(img)
        self.after_label.configure(image=self.after_img, text="")

    def run(self):
        inp = self.input_path.get().strip()
        if not inp:
            messagebox.showinfo("message", "Please select the image or folder.")
            return
        self.status.set("Processing...")
        threading.Thread(target=self._run_worker, args=(inp,), daemon=True).start()

    def _run_worker(self, inp):
        try:
            OUTPUTS.mkdir(exist_ok=True)
            cmd = [sys.executable, str(ROOT / "infer.py"), "--input", inp, "--output", str(OUTPUTS)]
            ckpt = self.checkpoint_path.get().strip()
            if ckpt:
                cmd += ["--checkpoint", ckpt]
            result = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
            if result.returncode != 0:
                self.status.set("Runtime Failure!")
                messagebox.showerror("خطا", result.stderr or result.stdout)
                return
            if Path(inp).is_file():
                out = OUTPUTS / f"{Path(inp).stem}_enhanced.png"
                if out.exists():
                    self.root.after(0, lambda: self.show_after(out))
            self.status.set("Done. Outputs saved in outputs.")
        except Exception as exc:
            self.status.set("Error")
            messagebox.showerror("Error", str(exc))


if __name__ == "__main__":
    root = Tk()
    root.geometry("980x560")
    App(root)
    root.mainloop()

