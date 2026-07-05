"""
MNIST Digit Drawing Tool
Draw a digit, click Submit to get a 784-float pixel vector (28x28, 0.0-1.0).
Works on Retina/HiDPI displays. Requires Python 3 + tkinter only.
"""

import tkinter as tk
from tkinter import messagebox
import math


import numpy as np
from NN import Layer, Activation, Network
from tensorflow import keras


np.random.seed(42)


# Define the Model Architecture
net = Network(
    input_size=784,
    layers=[
        Layer.Dense(128, activation=Activation("relu")),
        Layer.Dense(64,  activation=Activation("relu")),
        Layer.Dense(10,  activation=Activation("softmax")),
    ],
    loss="cross_entropy"
)

# Load the pretrained model
net.load("mnist_v2")
keras_model = keras.models.load_model("mnist_cnn.keras")


net.summary()
print()

PIXEL_COUNT  = 28
CELL_SIZE    = 16        # logical px per grid cell (28*16 = 448)
BRUSH_R      = 1.6       # brush radius in grid-cell units
PREVIEW_CELL = 8         # preview px per grid cell (28*8 = 224)

BG       = "#0d0d0d"
FG       = "#ffffff"
ACCENT   = "#4f9cf9"
C_CLEAR  = "#e05c5c"
C_SUBMIT = "#4caf7d"
FONT     = ("Consolas", 11)
FONT_B   = ("Consolas", 13, "bold")


class DigitDrawer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("MNIST Digit Drawer")
        self.configure(bg=BG)

        # Detect HiDPI/Retina scale and shrink logical sizes accordingly
        try:
            dpi   = self.winfo_fpixels("1i")
            scale = max(1, min(round(dpi / 96), 2))
        except Exception:
            scale = 1

        self.cell  = max(1, int(CELL_SIZE    // scale))
        self.pcell = max(1, int(PREVIEW_CELL // scale))
        self.cpx   = int(self.cell  * PIXEL_COUNT)
        self.ppx   = int(self.pcell * PIXEL_COUNT)

        self.resizable(False, False)
        self.grid_values = [[0.0] * PIXEL_COUNT for _ in range(PIXEL_COUNT)]
        self.last_cell   = None
        self.result_data = None

        self._build_ui()
        self._clear_grid()

    # -------------------------------------------------------------------------
    def _build_ui(self):
        cpx = self.cpx
        ppx = self.ppx

        # Title row
        tk.Label(self, text="MNIST Digit Drawer", font=FONT_B,
                 bg=BG, fg=ACCENT).pack(pady=(14, 0))
        tk.Label(self, text="Draw a single digit  |  28 x 28 output",
                 font=FONT, bg=BG, fg="#888").pack(pady=(2, 8))

        # ---- side-by-side: [drawing canvas]  [preview] ----------------------
        side_row = tk.Frame(self, bg=BG)
        side_row.pack(padx=14)

        # Drawing canvas (left)
        border = tk.Frame(side_row, bg=ACCENT, width=cpx + 4, height=cpx + 4)
        border.pack_propagate(False)
        border.pack(side="left")

        self.canvas = tk.Canvas(border, width=cpx, height=cpx,
                                bg=BG, cursor="crosshair",
                                highlightthickness=0, bd=0)
        self.canvas.place(x=2, y=2)

        self.canvas.bind("<ButtonPress-1>",   self._on_press)
        self.canvas.bind("<B1-Motion>",       self._on_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)

        # Preview panel (right)
        right = tk.Frame(side_row, bg=BG)
        right.pack(side="left", padx=(12, 0), anchor="n", pady=(0, 0))

        tk.Label(right, text="28 x 28", font=FONT, bg=BG, fg="#555").pack()
        tk.Label(right, text="preview", font=FONT, bg=BG, fg="#555").pack(pady=(0, 4))

        pborder = tk.Frame(right, bg="#333", width=ppx + 2, height=ppx + 2)
        pborder.pack_propagate(False)
        pborder.pack()

        self.preview = tk.Canvas(pborder, width=ppx, height=ppx,
                                 bg=BG, highlightthickness=0, bd=0)
        self.preview.place(x=1, y=1)

        # ---- buttons --------------------------------------------------------
        btn_row = tk.Frame(self, bg=BG)
        btn_row.pack(pady=12)

        tk.Button(btn_row, text="Clear", font=FONT,
                  bg=C_CLEAR, fg=FG, activebackground="#c04040",
                  activeforeground=FG, relief="flat", bd=0,
                  padx=18, pady=8, cursor="hand2",
                  command=self._clear_grid).pack(side="left", padx=8)

        tk.Button(btn_row, text="Submit", font=FONT,
                  bg=C_SUBMIT, fg=FG, activebackground="#3a8f60",
                  activeforeground=FG, relief="flat", bd=0,
                  padx=18, pady=8, cursor="hand2",
                  command=self._submit).pack(side="left", padx=8)

        # ---- status ---------------------------------------------------------
        self.status = tk.StringVar(value="Draw a digit above, then click Submit.")
        tk.Label(self, textvariable=self.status, font=FONT,
                 bg=BG, fg="#aaa", wraplength=cpx + ppx + 16).pack(pady=(0, 10))

    # -------------------------------------------------------------------------
    def _on_press(self, event):
        self.last_cell = self._xy_to_cell(event.x, event.y)
        self._paint_cell(*self.last_cell)

    def _on_drag(self, event):
        cell = self._xy_to_cell(event.x, event.y)
        if self.last_cell and cell != self.last_cell:
            self._paint_line(self.last_cell, cell)
        self.last_cell = cell

    def _on_release(self, _event):
        self.last_cell = None

    def _xy_to_cell(self, x, y):
        col = int(max(0, min(x, self.cpx - 1)) / self.cell)
        row = int(max(0, min(y, self.cpx - 1)) / self.cell)
        return (min(col, PIXEL_COUNT - 1), min(row, PIXEL_COUNT - 1))

    # -------------------------------------------------------------------------
    def _paint_cell(self, col, row):
        r  = BRUSH_R
        c0 = max(0, int(col - r - 1))
        c1 = min(PIXEL_COUNT - 1, int(col + r + 1))
        r0 = max(0, int(row - r - 1))
        r1 = min(PIXEL_COUNT - 1, int(row + r + 1))
        for gy in range(r0, r1 + 1):
            for gx in range(c0, c1 + 1):
                dist  = math.hypot(gx + 0.5 - (col + 0.5),
                                   gy + 0.5 - (row + 0.5))
                alpha = max(0.0, 1.0 - max(0.0, dist - r * 0.4) / (r * 0.6))
                self.grid_values[gy][gx] = min(
                    1.0, self.grid_values[gy][gx] + alpha * 0.85)
        self._redraw_canvas()
        self._redraw_preview()

    def _paint_line(self, cell0, cell1):
        c0, r0 = cell0
        c1, r1 = cell1
        dc, dr = c1 - c0, r1 - r0
        steps  = max(abs(dc), abs(dr), 1)
        for i in range(steps + 1):
            t  = i / steps
            gc = min(max(round(c0 + dc * t), 0), PIXEL_COUNT - 1)
            gr = min(max(round(r0 + dr * t), 0), PIXEL_COUNT - 1)
            self._paint_cell(gc, gr)

    # -------------------------------------------------------------------------
    def _redraw_canvas(self):
        cell = self.cell
        self.canvas.delete("px")
        for gy in range(PIXEL_COUNT):
            for gx in range(PIXEL_COUNT):
                v = self.grid_values[gy][gx]
                if v < 0.01:
                    continue
                g   = int(v * 255)
                col = f"#{g:02x}{g:02x}{g:02x}"
                x0, y0 = gx * cell, gy * cell
                self.canvas.create_rectangle(
                    x0, y0, x0 + cell, y0 + cell,
                    fill=col, outline="", tags="px")

    def _redraw_preview(self):
        pc = self.pcell
        self.preview.delete("all")
        for gy in range(PIXEL_COUNT):
            for gx in range(PIXEL_COUNT):
                v   = self.grid_values[gy][gx]
                g   = int(v * 255)
                col = f"#{g:02x}{g:02x}{g:02x}"
                x0, y0 = gx * pc, gy * pc
                self.preview.create_rectangle(
                    x0, y0, x0 + pc, y0 + pc,
                    fill=col, outline="")

    # -------------------------------------------------------------------------
    def _clear_grid(self):
        cpx  = self.cpx
        cell = self.cell
        self.grid_values = [[0.0] * PIXEL_COUNT for _ in range(PIXEL_COUNT)]
        self.canvas.delete("all")
        for i in range(0, cpx + 1, cell):
            self.canvas.create_line(i, 0, i, cpx, fill="#1c1c1c")
            self.canvas.create_line(0, i, cpx, i, fill="#1c1c1c")
        self._redraw_preview()
        self.status.set("Canvas cleared. Draw a digit above, then click Submit.")
        self.result_data = None

    def _submit(self):
        flat = [self.grid_values[r][c]
                for r in range(PIXEL_COUNT) for c in range(PIXEL_COUNT)]
        if max(flat) < 0.05:
            messagebox.showwarning("Empty canvas",
                                   "Please draw a digit before submitting.")
            return
        self.result_data = flat
        print("\n" + "=" * 60)
        print("Submitted! 784-element pixel vector (0.0 - 1.0):")
        print("-" * 60)
        nz  = sum(1 for v in flat if v > 0.01)
        avg = sum(flat) / len(flat)
        self.status.set(
            f"Submitted!  Non-zero pixels: {nz}/784  |  "
            f"Mean brightness: {avg:.4f}  |  Full vector in console.")

        ############## Predicting using the trained model ###########
        x = np.array(app.result_data, dtype=np.float32).reshape(1, -1)
        x_keras = np.array(app.result_data, dtype=np.float32).reshape(1, 28, 28, 1)

        prediction = net.predict(x)
        prediction_keras = keras_model.predict(x_keras)
        print(f"Predicted Digit using our ANN : {prediction}")
        print(f"Predicting Digit using the trained model with Keras: {np.argmax(prediction_keras, axis=1)}")

    def get_result(self):
        """Returns last submitted pixel list (784 floats) or None."""
        return self.result_data


if __name__ == "__main__":
    app = DigitDrawer()
    app.mainloop()
