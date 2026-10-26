"""Small optional Tk presentation helpers; no data access or plotting changes."""
from __future__ import annotations

import tkinter as tk
from tkinter import font as tkfont, ttk

BACKGROUND = "#f6f7f9"
PANEL = "#ffffff"
TEXT = "#242b38"
MUTED = "#788293"
ACCENT = "#386ce4"
BORDER = "#e5e8ee"
SIDEBAR = "#171d28"


def configure_theme(root):
    """Apply a light scientific-tool theme without changing Matplotlib settings."""
    style = ttk.Style(root)
    if "clam" in style.theme_names():
        style.theme_use("clam")
    root.configure(background=BACKGROUND)
    base = tkfont.nametofont("TkDefaultFont", root=root)
    family, size = base.actual("family"), base.actual("size")
    style.configure(".", font="TkDefaultFont", foreground=TEXT)
    style.configure("TFrame", background=PANEL)
    style.configure("App.TFrame", background=BACKGROUND)
    style.configure("Panel.TFrame", background=PANEL)
    style.configure("Header.TFrame", background=PANEL)
    style.configure("Sidebar.TFrame", background=SIDEBAR)
    style.configure("TLabel", background=PANEL, foreground=TEXT)
    style.configure("Header.TLabel", background=PANEL, foreground=TEXT)
    style.configure("Sidebar.TLabel", background=SIDEBAR, foreground="#b7c1d1")
    style.configure("Brand.TLabel", background=SIDEBAR, foreground="#ffffff",
                    font=(family, max(size + 8, 20), "bold"))
    style.configure("Sidebar.Muted.TLabel", background=SIDEBAR, foreground="#7e8b9f",
                    font=(family, max(9, size - 1)))
    style.configure("Header.Title.TLabel", background=PANEL, foreground=TEXT,
                    font=(family, max(size + 3, 15), "bold"))
    style.configure("Header.Muted.TLabel", background=PANEL, foreground=MUTED)
    style.configure("Section.Muted.TLabel", background=PANEL, foreground=MUTED)
    style.configure("Welcome.TLabel", background=PANEL, foreground=TEXT,
                    font=(family, max(size + 8, 20), "bold"))
    style.configure("Badge.TLabel", background="#edf2fc", foreground="#4d6592",
                    padding=(8, 4), font=(family, max(9, size - 1)))
    style.configure("Muted.TLabel", background=BACKGROUND, foreground=MUTED)
    style.configure("Title.TLabel", background=BACKGROUND, foreground=TEXT,
                    font=(family, max(size + 5, 15), "bold"))
    style.configure("Section.TLabel", background=PANEL, foreground=TEXT,
                    font=(family, size, "bold"))
    style.configure("TLabelframe", background=PANEL, bordercolor=BORDER, relief="solid")
    style.configure("TLabelframe.Label", background=PANEL, foreground=MUTED)
    style.configure("TButton", padding=(11, 6), background="#f0f2f6", borderwidth=0,
                    relief="flat", bordercolor=PANEL, lightcolor=PANEL,
                    darkcolor=PANEL, focuscolor=ACCENT)
    style.map("TButton", background=[("disabled", "#f3f4f6"), ("pressed", "#e1e6ee"),
                                      ("active", "#e8edf5")],
              foreground=[("disabled", "#a1a9b6"), ("!disabled", TEXT)])
    style.configure("Accent.TButton", background=ACCENT, foreground="white",
                    bordercolor=ACCENT, padding=(12, 6))
    style.map("Accent.TButton", background=[("disabled", "#dce4ef"),
                                              ("pressed", "#1e40af"), ("active", "#1d4ed8")],
              foreground=[("disabled", MUTED), ("!disabled", "white")])
    style.configure("Section.TButton", anchor="w", padding=(8, 6), background=PANEL,
                    borderwidth=0, relief="flat")
    style.map("Section.TButton", background=[("active", "#f3f5f8"), ("!active", PANEL)])
    style.configure("Nav.TButton", anchor="w", padding=(14, 11), background=SIDEBAR,
                    foreground="#aeb9cb", borderwidth=0, relief="flat", focuscolor="#8598bc")
    style.map("Nav.TButton", background=[("active", "#232c3b"), ("!active", SIDEBAR)],
              foreground=[("disabled", "#586274"), ("active", "#ffffff")])
    style.configure("Selected.Nav.TButton", background="#28344a", foreground="#ffffff")
    style.map("Selected.Nav.TButton", background=[("active", "#303e56"),
                                                   ("!active", "#28344a")],
              foreground=[("!disabled", "#ffffff")])
    style.configure("TCheckbutton", background=PANEL, padding=(3, 3))
    style.map("TCheckbutton", background=[("active", PANEL)])
    style.configure("TEntry", fieldbackground=PANEL, bordercolor=BORDER, padding=5)
    style.configure("TCombobox", fieldbackground=PANEL, bordercolor=BORDER, padding=4)
    style.map("TCombobox", fieldbackground=[("readonly", PANEL)])
    style.configure("TNotebook", background=BACKGROUND, borderwidth=0)
    style.configure("TNotebook.Tab", padding=(16, 9), background=BACKGROUND)
    style.map("TNotebook.Tab", background=[("selected", PANEL)],
              foreground=[("selected", ACCENT)])
    style.configure("Workspace.TNotebook", background=PANEL, borderwidth=0)
    style.layout("Workspace.TNotebook.Tab", [])
    style.configure("Treeview", background=PANEL, fieldbackground=PANEL,
                    foreground=TEXT, rowheight=max(26, base.metrics("linespace") + 10),
                    bordercolor=BORDER)
    style.map("Treeview", background=[("selected", "#dce9ff")],
              foreground=[("selected", TEXT)])
    style.configure("Treeview.Heading", background="#eef3f9", foreground=TEXT,
                    padding=(7, 7), font=(family, size, "bold"))
    style.configure("ToolTip.TLabel", background="#172b46", foreground="white")
    # Flat surfaces remove the inherited bevels from the classic clam theme.
    style.configure("Workspace.TNotebook", bordercolor=PANEL, lightcolor=PANEL, darkcolor=PANEL)
    style.configure("TLabelframe", borderwidth=0, relief="flat", lightcolor=PANEL, darkcolor=PANEL)
    for name in ("TEntry", "TCombobox", "TSpinbox"):
        style.configure(name, borderwidth=1, relief="solid", lightcolor=BORDER,
                        darkcolor=BORDER, bordercolor=BORDER, arrowcolor=MUTED)
    style.configure("Treeview", borderwidth=0, relief="flat", lightcolor=PANEL, darkcolor=PANEL)
    style.configure("Treeview.Heading", relief="flat", borderwidth=0,
                    lightcolor="#eef2f8", darkcolor="#eef2f8", bordercolor="#eef2f8")
    for name in ("Vertical.TScrollbar", "Horizontal.TScrollbar"):
        style.configure(name, background="#cbd2dd", troughcolor="#f4f6f9", bordercolor=PANEL,
                        lightcolor="#cbd2dd", darkcolor="#cbd2dd", arrowcolor="#7d8796", arrowsize=11)
    style.configure("TScale", background=PANEL, troughcolor="#e5ebf6", bordercolor=PANEL,
                    lightcolor=PANEL, darkcolor=PANEL)
    return style


class RoundedButton(tk.Canvas):
    """A keyboard-accessible rounded action button using only native Canvas.

    ``text``, ``state`` and ``command`` support configure/cget; ``invoke`` is a
    no-op while disabled. Set ``background`` to the containing surface colour.
    """

    def __init__(self, master, text="", command=None, width=None, height=36,
                 variant="primary", **kwargs):
        self._text, self._command = str(text), command
        self._state = kwargs.pop("state", "normal")
        self._variant, self._auto_width = variant, width is None
        self._font = tkfont.Font(root=master, font=kwargs.pop("font", "TkDefaultFont"))
        self._font.configure(weight="bold")
        self._hover = self._pressed = self._focused = False
        background = kwargs.pop("background", kwargs.pop("bg", PANEL))
        super().__init__(master, width=width or max(96, self._font.measure(self._text) + 32),
                         height=height, background=background, highlightthickness=0,
                         borderwidth=0, takefocus=self._state != "disabled", **kwargs)
        self.bind("<Configure>", self._draw)
        self.bind("<Enter>", lambda event: self._appearance(hover=True))
        self.bind("<Leave>", lambda event: self._appearance(hover=False))
        self.bind("<FocusIn>", lambda event: self._appearance(focused=True))
        self.bind("<FocusOut>", lambda event: self._appearance(focused=False, pressed=False))
        self.bind("<ButtonPress-1>", self._press)
        self.bind("<ButtonRelease-1>", self._release)
        for key in ("space", "Return"):
            self.bind(f"<KeyPress-{key}>", self._press)
            self.bind(f"<KeyRelease-{key}>", self._release)
        self.configure(state=self._state)

    def _appearance(self, **values):
        for name, value in values.items():
            setattr(self, "_" + name, value)
        self._draw()

    def _press(self, event=None):
        if self._state != "disabled":
            self.focus_set()
            self._appearance(pressed=True)
            if getattr(event, "keysym", "") == "Return":
                self.invoke()
        return "break"

    def _release(self, event):
        pressed = self._pressed
        self._appearance(pressed=False)
        keyboard = getattr(event, "keysym", "") in ("space", "Return")
        inside = keyboard or (0 <= event.x < self.winfo_width() and 0 <= event.y < self.winfo_height())
        if pressed and inside and getattr(event, "keysym", "") != "Return":
            self.invoke()
        return "break"

    def _pill(self, x, y, width, height, radius=10, **kwargs):
        r = min(radius, max(0, width / 2), max(0, height / 2))
        return self.create_polygon(x+r, y, x+width-r, y, x+width, y,
            x+width, y+r, x+width, y+height-r, x+width, y+height,
            x+width-r, y+height, x+r, y+height, x, y+height,
            x, y+height-r, x, y+r, x, y, smooth=True, splinesteps=24, **kwargs)

    def _draw(self, _event=None):
        width, height = self.winfo_width(), self.winfo_height()
        if width <= 1:
            width = int(float(super().cget("width")))
        if height <= 1:
            height = int(float(super().cget("height")))
        self.delete("all")
        disabled = self._state == "disabled"
        if disabled:
            face, ink = "#e9ecf1", "#a0a8b5"
        elif self._variant == "primary":
            face = "#2a53b5" if self._pressed else ("#2f60d2" if self._hover else ACCENT)
            ink = "#ffffff"
        else:
            face = "#e1e6ee" if self._pressed else ("#e8edf5" if self._hover else "#f0f2f6")
            ink = TEXT
        if self._focused and not disabled:
            self._pill(0, 0, width, height, radius=12, fill="#b5c9f7", outline="")
        self._pill(2, 2, max(0, width-4), max(0, height-4), fill=face, outline="")
        self.create_text(width/2, height/2, text=self._text, fill=ink, font=self._font)

    def configure(self, cnf=None, **kwargs):
        options = dict(cnf or {}, **kwargs)
        if "text" in options:
            self._text = str(options.pop("text"))
            if self._auto_width:
                options.setdefault("width", max(96, self._font.measure(self._text) + 32))
        if "command" in options:
            self._command = options.pop("command")
        if "state" in options:
            state = options.pop("state")
            if state not in ("normal", "disabled"):
                raise ValueError("state must be normal or disabled")
            self._state = state
            if state == "disabled":
                self._pressed = self._hover = False
            options.update(takefocus=state != "disabled", cursor="arrow" if state == "disabled" else "hand2")
        if options:
            super().configure(**options)
        self._draw()

    config = configure

    def cget(self, key):
        if key in ("text", "state", "command"):
            return {"text": self._text, "state": self._state, "command": self._command}[key]
        return super().cget(key)

    def invoke(self):
        if self._state != "disabled" and self._command is not None:
            return self._command()


def initial_geometry(root):
    """Return a centered geometry and minimum size that fit the current screen."""
    screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
    width, height = min(1360, max(1, screen_w - 64)), min(860, max(1, screen_h - 96))
    x, y = max(0, (screen_w - width) // 2), max(0, (screen_h - height) // 2)
    return f"{width}x{height}+{x}+{y}", (min(1000, width), min(660, height))


class ScrollableFrame(ttk.Frame):
    """A width-fitted control panel; wheel bindings belong only to its subtree."""

    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(self, background=PANEL, highlightthickness=0,
                                borderwidth=0, yscrollincrement=24)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.scrollbar.grid(row=0, column=1, sticky="ns")
        self.body = ttk.Frame(self.canvas, style="Panel.TFrame")
        self._window = self.canvas.create_window(0, 0, window=self.body, anchor="nw")
        self._tag = "CessLocalWheel" + str(id(self))
        self._refresh_after = None
        self._bindings = {}
        for sequence in ("<MouseWheel>", "<Button-4>", "<Button-5>"):
            self._bindings[sequence] = self.bind_class(self._tag, sequence, self._wheel)
        self.body.bind("<Configure>", self._layout, add="+")
        self.canvas.bind("<Configure>", self._resize, add="+")
        self.bind("<Destroy>", self._cleanup, add="+")
        self._refresh_after = self.after_idle(self.refresh)

    def refresh(self):
        """Include newly constructed children in this panel's local wheel tag."""
        self._refresh_after = None
        pending = [self]
        while pending:
            widget = pending.pop()
            tags = widget.bindtags()
            if self._tag not in tags:
                widget.bindtags((tags[0], self._tag, *tags[1:]))
            pending.extend(widget.winfo_children())

    def _layout(self, _event=None):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
        if self._refresh_after is None:
            self._refresh_after = self.after_idle(self.refresh)

    def _resize(self, event):
        self.canvas.itemconfigure(self._window, width=max(1, event.width))

    def _wheel(self, event):
        widget = event.widget
        parent = widget
        while parent is not None and parent is not self:
            parent = getattr(parent, "master", None)
        if parent is not self:
            return
        x, y = self.winfo_pointerxy()
        if not (self.winfo_rootx() <= x < self.winfo_rootx() + self.winfo_width()
                and self.winfo_rooty() <= y < self.winfo_rooty() + self.winfo_height()):
            return
        if isinstance(widget, (tk.Text, tk.Listbox, ttk.Treeview)) and widget.yview() != (0., 1.):
            return  # A nested data preview keeps its own native scrolling.
        if isinstance(widget, (ttk.Combobox, tk.Spinbox, ttk.Spinbox)):
            return
        if self.canvas.yview() == (0., 1.):
            return
        if getattr(event, "num", None) in (4, 5):
            steps = -1 if event.num == 4 else 1
        else:
            delta = getattr(event, "delta", 0)
            if not delta:
                return
            divisor = 1 if self.tk.call("tk", "windowingsystem") == "aqua" else 120
            steps = -int(delta / divisor) or (-1 if delta > 0 else 1)
        self.canvas.yview_scroll(steps, "units")
        return "break"

    def _cleanup(self, event):
        if event.widget is not self:
            return
        if self._refresh_after is not None:
            self.after_cancel(self._refresh_after)
            self._refresh_after = None
        for sequence, command in self._bindings.items():
            self.unbind_class(self._tag, sequence)
            if command:
                self._root().deletecommand(command)
        self._bindings.clear()


class CollapsibleFrame(ttk.Frame):
    """A section whose child controls retain their state while hidden."""

    def __init__(self, master, text, expanded=False, **kwargs):
        super().__init__(master, **kwargs)
        self._text, self.expanded = text, bool(expanded)
        self.button = ttk.Button(self, style="Section.TButton", command=self.toggle)
        self.button.pack(fill="x")
        self.body = ttk.Frame(self, style="Panel.TFrame", padding=(6, 3, 6, 6))
        self._render()

    def _render(self):
        self.button.configure(text=("▾ " if self.expanded else "▸ ") + self._text)
        if self.expanded:
            self.body.pack(fill="both", expand=True)
        else:
            self.body.pack_forget()

    def toggle(self):
        self.expanded = not self.expanded
        self._render()


class ToolTip:
    """Delayed, non-interactive help attached to a single widget."""

    def __init__(self, widget, text, delay=500):
        self.widget, self.text, self.delay = widget, text, delay
        self._after = self._window = None
        self._bindings = {event: widget.bind(event, callback, add="+") for event, callback in (
            ("<Enter>", self._enter), ("<Leave>", self.hide),
            ("<ButtonPress>", self.hide), ("<Destroy>", self._destroy_event))}

    def _enter(self, _event=None):
        self.hide()
        self._after = self.widget.after(self.delay, self._show)

    def _show(self):
        self._after = None
        if not self.widget.winfo_exists():
            return
        self._window = window = tk.Toplevel(self.widget)
        window.wm_overrideredirect(True)
        ttk.Label(window, text=self.text, style="ToolTip.TLabel", padding=(9, 6),
                  wraplength=420, justify="left").pack()
        window.update_idletasks()
        x = min(self.widget.winfo_rootx(), max(0, window.winfo_screenwidth() - window.winfo_reqwidth() - 8))
        y = min(self.widget.winfo_rooty() + self.widget.winfo_height() + 6,
                max(0, window.winfo_screenheight() - window.winfo_reqheight() - 8))
        window.wm_geometry(f"+{max(0, x)}+{max(0, y)}")

    def hide(self, _event=None):
        if self._after is not None:
            self.widget.after_cancel(self._after)
            self._after = None
        if self._window is not None:
            self._window.destroy()
            self._window = None

    def _destroy_event(self, event):
        if event.widget is self.widget:
            self.destroy()

    def destroy(self):
        self.hide()
        for event, command in self._bindings.items():
            if command:
                self.widget.unbind(event, command)
        self._bindings.clear()
