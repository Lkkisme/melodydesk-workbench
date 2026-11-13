"""MelodyDesk: a local Windows desktop prototype, no external audio uploads."""
import copy
from pathlib import Path
import os
import sys
import queue
import threading
import time
import traceback
import winsound

# The host's Tcl resolves some absolute Windows paths incorrectly. Relative
# script-library paths work with the project's copied runtime libraries.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from runtime_paths import get_install_root, get_work_root
_local_tcl = get_work_root() / 'runtime' / 'tcl'
if _local_tcl.is_dir():
    os.chdir(get_install_root())
    for _env_k, _sub in (('TCL_LIBRARY', 'tcl8.6'), ('TK_LIBRARY', 'tk8.6')):
        _target = _local_tcl / _sub
        try:
            os.environ[_env_k] = os.path.relpath(_target).replace('\\', '/')
        except ValueError:
            os.environ[_env_k] = str(_target.resolve()).replace('\\', '/')

import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from music import Project, Note, NAMES, numbered, pitch_name, parse_jianpu, quantize, midi_bytes, text_score
import engine

BG = '#10151f'
PANEL = '#192231'
FG = '#eaf0f8'
MUTED = '#9cacc2'
ACCENT = '#60dbc0'
AMBER = '#f6c177'
CONFIDENCE_THRESHOLD = 0.7


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('MelodyDesk · 歌曲旋律工作台')
        self.geometry('1180x850')
        self.minsize(1000, 720)
        self.configure(bg=BG)
        self.project = Project()
        self.source = ''
        self.source_duration = 0
        self.source_probed = False
        self.clip_end = 0
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.busy = False
        self.active_job_kind = None
        self.job_token = 0
        self._destroyed = False
        self.dirty = False
        self.undo_stack = []
        self.play_started = None
        self.play_duration = 0
        self.play_kind = 'grid'
        self.play_pending = False
        self.play_token = 0
        self.pending_probe = 0
        self.pending_close = False
        self.status = tk.StringVar(value='导入一首歌曲，或切换到「手动简谱」。')
        self.filename = tk.StringVar(value='尚未选择歌曲')
        self.start = tk.StringVar(value='0')
        self.duration = tk.StringVar(value='30')
        self.bpm = tk.StringVar(value='120')
        self.meter = tk.StringVar(value='4/4')
        self.tonic = tk.StringVar(value='C')
        self.origin = tk.StringVar(value='0')
        self.pitch = tk.StringVar(value='60')
        self.note_start = tk.StringVar(value='0')
        self.note_duration = tk.StringVar(value='1')
        self.timing_choice = tk.StringVar(value='原演唱节奏')
        self.model_choice = tk.StringVar(value='稳定音高（默认）')
        self.vocal_only = tk.BooleanVar(value=True)
        self.full_song = tk.BooleanVar(value=False)
        self.rhythm_mode = tk.StringVar(value='自动节奏')
        self.zoom = tk.StringVar(value='1')
        self.drag = None
        self.looping = False
        self.play_offset = 0
        self._style()
        self._build()
        self.poll_timer = self.after(120, self._poll)
        self.withdraw()
        self.show_timer = self.after(200, self.deiconify)
        self.protocol('WM_DELETE_WINDOW', self.close)
        self.report_callback_exception = self.callback_error
        self.bind('<Control-s>', lambda e: self.save())
        self.bind('<Control-z>', lambda e: self.undo())
        missing = engine.dependencies()
        if missing:
            self.status.set('运行组件未就绪：' + '、'.join(missing) + '。可先使用手动简谱。')

    def _style(self):
        self.option_add('*Font', ('Microsoft YaHei UI', 10))
        s = ttk.Style(self)
        s.theme_use('clam')
        s.configure('.', background=BG, foreground=FG, font=('Microsoft YaHei UI', 10))
        s.configure('TFrame', background=BG)
        s.configure('TLabel', background=BG, foreground=FG)
        s.configure('Muted.TLabel', foreground=MUTED)
        s.configure('Title.TLabel', font=('Microsoft YaHei UI', 22, 'bold'))
        s.configure('TButton', background='#263449', foreground=FG, borderwidth=0, padding=(12, 8))
        s.map('TButton', background=[('active', '#364d65'), ('disabled', '#202935')], foreground=[('disabled', '#65758a')])
        s.configure('Accent.TButton', background=ACCENT, foreground=BG)
        s.map('Accent.TButton', background=[('active', '#91ead8'), ('disabled', '#314640')])
        s.configure('TEntry', fieldbackground=PANEL, foreground=FG, insertcolor=FG, padding=5)
        s.configure('TCombobox', fieldbackground=PANEL, background=PANEL, foreground=FG, padding=5)
        s.map('TCombobox', fieldbackground=[('readonly', PANEL)], foreground=[('readonly', FG)])
        s.configure('TNotebook', background=BG, borderwidth=0)
        s.configure('TNotebook.Tab', padding=(18, 10), background=PANEL, foreground=MUTED)
        s.map('TNotebook.Tab', background=[('selected', '#263449')], foreground=[('selected', ACCENT)])
        s.configure('Treeview', background=PANEL, fieldbackground=PANEL, foreground=FG, rowheight=30, borderwidth=0)
        s.configure('Treeview.Heading', background='#263449', foreground=MUTED, padding=6)
        s.map('Treeview', background=[('selected', '#30566b')], foreground=[('selected', '#ffffff')])
        s.configure('Horizontal.TProgressbar', troughcolor=PANEL, background=ACCENT, borderwidth=0)

    def _build(self):
        main = ttk.Frame(self, padding=24)
        main.pack(fill='both', expand=True)
        header = ttk.Frame(main)
        header.pack(fill='x')
        ttk.Label(header, text='MelodyDesk', style='Title.TLabel').pack(side='left')
        ttk.Label(header, text='  /  歌曲旋律工作台', style='Muted.TLabel').pack(side='left', pady=(8, 0))
        ttk.Button(header, text='打开项目', command=self.open_project).pack(side='right', padx=(8, 0))
        ttk.Button(header, text='保存项目', command=self.save).pack(side='right')
        ttk.Button(header, text='吉他编配', command=self.guitar_dialog).pack(side='right', padx=8)
        ttk.Label(main, text='01  提取旋律     →     02  试听校对     →     03  导出编配数据', style='Muted.TLabel').pack(anchor='w', pady=(10, 16))

        inputs = ttk.Notebook(main)
        inputs.pack(fill='x')
        audio_tab = ttk.Frame(inputs, padding=14)
        manual_tab = ttk.Frame(inputs, padding=14)
        inputs.add(audio_tab, text='歌曲音频')
        inputs.add(manual_tab, text='手动简谱')
        row = ttk.Frame(audio_tab)
        row.pack(fill='x')
        self.choose_button = ttk.Button(row, text='＋ 选择完整歌曲', command=self.choose_audio)
        self.choose_button.pack(side='left')
        ttk.Label(row, textvariable=self.filename, style='Muted.TLabel', wraplength=700).pack(side='left', padx=14)
        row = ttk.Frame(audio_tab)
        row.pack(fill='x', pady=(12, 0))
        self._field(row, '开始 / 秒', self.start, 8)
        self._field(row, '长度 / 秒', self.duration, 8)
        ttk.Checkbutton(row,text='整首歌曲',variable=self.full_song).pack(side='left',padx=5)
        self.extract_button = ttk.Button(row, text='开始提取旋律', style='Accent.TButton', command=self.analyze)
        self.extract_button.pack(side='left', padx=(12, 8))
        self.cancel_button = ttk.Button(row, text='取消', command=self.cancel_job, state='disabled')
        self.cancel_button.pack(side='left')
        row = ttk.Frame(audio_tab)
        row.pack(fill='x', pady=(8, 0))
        ttk.Label(row, text='旋律方案').pack(side='left')
        ttk.Combobox(row, textvariable=self.model_choice, values=['稳定音高（默认）', '演唱音符模型（实验）'], state='readonly', width=25).pack(side='left', padx=8)
        ttk.Checkbutton(row, text='仅保留人声区域', variable=self.vocal_only).pack(side='left')
        ttk.Label(audio_tab, text='支持片段或整首歌曲（最长 60 分钟）。整曲处理较慢，可取消；音频在本机处理。',
                  style='Muted.TLabel').pack(anchor='w', pady=(10, 0))
        ttk.Label(manual_tab, text="格式：音级:拍数，例如 1:1 2:0.5 3:0.5 0:1；' 升八度，, 降八度。拍数以四分音符为单位。",
                  style='Muted.TLabel').pack(anchor='w')
        row = ttk.Frame(manual_tab)
        row.pack(fill='x', pady=(10, 0))
        self.manual_text = tk.Text(row, height=2, bg=PANEL, fg=FG, insertbackground=FG, relief='flat', font=('Consolas', 13), wrap='word')
        self.manual_text.pack(side='left', fill='x', expand=True)
        self.manual_text.insert('1.0', '1:1 2:1 3:1 5:1 | 6:1 5:1 3:1 0:1')
        ttk.Button(row, text='生成旋律', command=self.from_manual).pack(side='left', padx=(12, 0))

        row = ttk.Frame(main)
        row.pack(fill='x', pady=(14, 8))
        self._field(row, '速度 BPM', self.bpm, 7)
        ttk.Label(row, text='拍号').pack(side='left', padx=(12, 6))
        ttk.Combobox(row, textvariable=self.meter, values=['2/4', '3/4', '4/4', '6/8'], state='readonly', width=5).pack(side='left')
        ttk.Label(row, text='简谱 1=').pack(side='left', padx=(12, 6))
        ttk.Combobox(row, textvariable=self.tonic, values=NAMES, state='readonly', width=4).pack(side='left')
        self._field(row, '拍点偏移 / 秒', self.origin, 7)
        ttk.Button(row, text='应用设置', command=self.apply_settings).pack(side='left', padx=8)
        ttk.Button(row, text='按原始时间重排', command=self.requantize).pack(side='left')
        toolbar = ttk.Frame(main)
        toolbar.pack(fill='x', pady=(0, 6))
        ttk.Combobox(toolbar, textvariable=self.rhythm_mode, values=['自动节奏', '普通节奏', '三连音'], state='readonly', width=9).pack(side='left')
        ttk.Button(toolbar, text='整理全段', command=lambda: self.organize_score(False)).pack(side='left', padx=4)
        ttk.Button(toolbar, text='整理选中', command=lambda: self.organize_score(True)).pack(side='left')
        ttk.Button(toolbar, text='归并为一个音', command=self.collapse_selected).pack(side='left', padx=4)
        ttk.Button(toolbar, text='恢复识别参考', command=self.restore_reference).pack(side='left')
        ttk.Button(toolbar, text='歌词', command=self.lyrics_dialog).pack(side='left',padx=4)
        ttk.Label(toolbar, text='放大').pack(side='left', padx=5)
        zoom = ttk.Combobox(toolbar, textvariable=self.zoom, values=['1','2','4','8'], state='readonly', width=3)
        zoom.pack(side='left'); zoom.bind('<<ComboboxSelected>>', lambda e:self.draw_roll())

        self.roll = tk.Canvas(main, bg=PANEL, height=160, highlightthickness=0)
        self.roll.pack(fill='x')
        self.roll.bind('<Configure>', lambda e: self.draw_roll())
        self.roll.bind('<Button-1>', self.pick_roll)
        self.roll.bind('<B1-Motion>', self.drag_roll)
        self.roll.bind('<ButtonRelease-1>', self.release_roll)
        horizontal = ttk.Scrollbar(main, orient='horizontal', command=self.roll.xview)
        horizontal.pack(fill='x')
        self.roll.configure(xscrollcommand=horizontal.set)
        self.stats = tk.StringVar(value='尚无旋律 · 识别后可点击音符校对')
        ttk.Label(main, textvariable=self.stats, style='Muted.TLabel').pack(anchor='w', pady=(6, 8))

        player = ttk.Frame(main)
        player.pack(fill='x', pady=(0, 12))
        ttk.Button(player, text='▶ 原曲片段', command=lambda: self.play('mix')).pack(side='left')
        ttk.Button(player, text='▶ 人声', command=lambda: self.play('vocals')).pack(side='left', padx=6)
        ttk.Button(player, text='▶ 旋律', command=lambda: self.play('melody')).pack(side='left')
        ttk.Combobox(player, textvariable=self.timing_choice, values=['原演唱节奏', '谱面节奏'], state='readonly', width=11).pack(side='left', padx=4)
        ttk.Button(player, text='▶ 原曲＋旋律', command=lambda: self.play('compare')).pack(side='left', padx=6)
        ttk.Button(player, text='停止', command=self.stop).pack(side='left')
        ttk.Button(player, text='循环选中谱面', command=self.loop_selected).pack(side='left', padx=4)
        ttk.Button(player, text='导出 MIDI', command=lambda: self.export('midi')).pack(side='right')
        ttk.Button(player, text='导出简谱文本', command=lambda: self.export('text')).pack(side='right', padx=6)

        table_frame = ttk.Frame(main)
        table_frame.pack(fill='both', expand=True)
        columns = ('index', 'degree', 'pitch', 'start', 'duration', 'raw', 'confidence')
        self.table = ttk.Treeview(table_frame, columns=columns, show='headings', height=5, selectmode='extended')
        for col, title, width in zip(columns, ['序号', '简谱', '音高', '起始拍', '时值 / 拍', '原曲时间 / 秒', '检查状态'], [60, 75, 95, 85, 85, 150, 160]):
            self.table.heading(col, text=title)
            self.table.column(col, width=width, anchor='center', minwidth=50)
        self.table.tag_configure('uncertain', foreground=AMBER)
        scroll = ttk.Scrollbar(table_frame, orient='vertical', command=self.table.yview)
        self.table.configure(yscrollcommand=scroll.set)
        self.table.pack(side='left', fill='both', expand=True)
        scroll.pack(side='right', fill='y')
        self.table.bind('<<TreeviewSelect>>', self.select_note)

        editor = ttk.Frame(main)
        editor.pack(fill='x', pady=(10, 8))
        self._field(editor, 'MIDI 音高', self.pitch, 5)
        self._field(editor, '起始拍', self.note_start, 6)
        self._field(editor, '时值', self.note_duration, 6)
        ttk.Button(editor, text='修改', command=self.edit_note).pack(side='left', padx=(8, 4))
        ttk.Button(editor, text='新增', command=self.add_note).pack(side='left', padx=4)
        ttk.Button(editor, text='拆分', command=self.split_note).pack(side='left', padx=4)
        ttk.Button(editor, text='合并', command=self.merge_notes).pack(side='left', padx=4)
        ttk.Button(editor, text='删除', command=self.delete_notes).pack(side='left', padx=4)
        ttk.Button(editor, text='撤销', command=self.undo).pack(side='left', padx=4)
        self.progress = ttk.Progressbar(main, maximum=100)
        self.progress.pack(fill='x', pady=(4, 8))
        ttk.Label(main, textvariable=self.status, style='Muted.TLabel', wraplength=1050).pack(anchor='w')

    def _field(self, parent, label, variable, width):
        ttk.Label(parent, text=label).pack(side='left', padx=(8, 6))
        ttk.Entry(parent, textvariable=variable, width=width).pack(side='left')

    def callback_error(self, kind, value, tb):
        engine.WORK.mkdir(parents=True, exist_ok=True)
        (engine.WORK / 'last-error.log').write_text(''.join(traceback.format_exception(kind, value, tb)), encoding='utf-8')
        messagebox.showerror('请检查', str(value), parent=self)

    def available(self):
        if self.busy:
            self.status.set('正在处理音频，请等待完成或先取消。')
            return False
        return True

    def confirm_replace(self):
        return not self.dirty or messagebox.askyesno('替换当前旋律', '当前修改尚未保存，继续将替换这些修改。是否继续？', parent=self)

    def _job(self, worker, kind):
        self.busy = True
        self.active_job_kind = kind
        self.job_token = getattr(self, 'job_token', 0) + 1
        my_job_token = self.job_token
        if kind == 'play':
            self.play_token = getattr(self, 'play_token', 0) + 1
            self.play_pending = True
        my_play_token = getattr(self, 'play_token', 0)
        self.cancel.clear()
        self.extract_button.configure(state='disabled')
        self.choose_button.configure(state='disabled')
        self.cancel_button.configure(state='normal' if kind in ('result','lyrics') else 'disabled')
        def task():
            try:
                result = worker()
                if my_job_token != getattr(self, 'job_token', 0):
                    return
                if kind == 'play':
                    if my_play_token != getattr(self, 'play_token', 0):
                        return
                    if isinstance(result, tuple) and len(result) == 3:
                        result = (*result, my_play_token)
                self.events.put((kind, result, my_job_token))
            except engine.Cancelled:
                if my_job_token == getattr(self, 'job_token', 0):
                    self.events.put(('cancelled', None, my_job_token))
            except Exception as exc:
                engine.WORK.mkdir(parents=True, exist_ok=True)
                (engine.WORK / 'last-error.log').write_text(traceback.format_exc(), encoding='utf-8')
                if my_job_token == getattr(self, 'job_token', 0):
                    self.events.put(('error', str(exc), my_job_token))
        threading.Thread(target=task, daemon=True).start()

    def _poll(self):
        try:
            try:
                while True:
                    item = self.events.get_nowait()
                    if isinstance(item, tuple) and len(item) == 3:
                        kind, value, event_job_token = item
                        if event_job_token != getattr(self, 'job_token', 0):
                            continue
                    else:
                        kind, value = item
                    if kind == 'progress':
                        self.progress['value'], text = value
                        self.status.set(text)
                        continue
                    if kind == 'play':
                        if isinstance(value, tuple) and len(value) == 4:
                            path, duration, mode, event_token = value
                            if event_token != getattr(self, 'play_token', 0) or not getattr(self, 'play_pending', False):
                                continue
                        else:
                            if not getattr(self, 'play_pending', False):
                                continue
                            path, duration, mode = value
                    self.busy = False
                    self.active_job_kind = None
                    if self.pending_close:
                        self.stop()
                        self.destroy()
                        return
                    self.extract_button.configure(state='normal')
                    self.choose_button.configure(state='normal')
                    self.cancel_button.configure(state='disabled')
                    if kind == 'result':
                        self.set_project(value)
                        self.dirty = True
                        self.status.set('提取完成。拍号与调号仍需确认；橙色音符建议优先试听检查。')
                    elif kind == 'probe':
                        path, length = value
                        self.source = path
                        self.source_duration = length
                        self.source_probed = True
                        self.filename.set(f'{Path(path).name}  ·  {int(length // 60):02}:{int(length % 60):02}')
                        self.start.set('0')
                        self.duration.set(str(round(min(30, length), 2)))
                        self.status.set('歌曲已加载。选择主唱清楚的片段，然后开始提取。')
                    elif kind == 'guitar':
                        self.status.set('吉他草稿已导出：' + str(value))
                        try:
                            os.startfile(str(value))
                        except OSError:
                            messagebox.showinfo('吉他谱已保存', str(value), parent=self)
                    elif kind == 'lyrics':
                        candidate = copy.deepcopy(self.project); candidate.lyrics = value; candidate.validate()
                        self.remember()
                        self.project = candidate
                        self.status.set(f'识别得到 {len(value)} 段歌词，需人工校对。请打开「歌词」修改。')
                    elif kind == 'play':
                        self.play_pending = False
                        flags = winsound.SND_FILENAME | winsound.SND_ASYNC | winsound.SND_NODEFAULT
                        if mode == 'loop':
                            flags |= winsound.SND_LOOP
                        try:
                            winsound.PlaySound(str(path), flags)
                        except Exception as exc:
                            self.looping = False
                            self.play_started = None
                            self.status.set('试听启动失败：' + str(exc))
                            messagebox.showerror('播放失败', str(exc), parent=self)
                            continue
                        self.looping = mode == 'loop'
                        self.play_started = time.monotonic()
                        self.play_duration = duration
                        self.play_kind = mode
                        self.status.set('正在循环选中谱面，点击「停止」结束。' if mode == 'loop' else '正在试听；「原曲＋旋律」使用原始演唱时间对齐。')
                    elif kind == 'error':
                        self.play_pending = False
                        self.status.set('处理未完成，可重试或更换片段。')
                        messagebox.showerror('处理失败', value, parent=self)
                    elif kind == 'cancelled':
                        self.play_pending = False
                        self.progress['value'] = 0
                        self.status.set('已取消，原有旋律保留。')
            except queue.Empty as exc:
                _ = exc
            if self.play_started is not None:
                elapsed = time.monotonic() - self.play_started
                if elapsed >= self.play_duration:
                    if self.looping:
                        self.play_started = time.monotonic()
                    else:
                        self.stop()
                else:
                    self.draw_roll(elapsed + self.play_offset)
        finally:
            if not getattr(self, '_destroyed', False) and self.winfo_exists():
                self.poll_timer = self.after(100, self._poll)

    def destroy(self):
        self._destroyed = True
        for timer in (getattr(self,'poll_timer',None),getattr(self,'show_timer',None)):
            if timer:
                try:
                    self.after_cancel(timer)
                except tk.TclError as exc:
                    _ = exc
        super().destroy()

    def choose_audio(self):
        if not self.available():
            return
        path = filedialog.askopenfilename(title='选择完整歌曲', filetypes=[('音频', '*.mp3 *.wav *.flac *.m4a *.ogg *.aac'), ('所有文件', '*.*')])
        if path:
            self.status.set('读取歌曲信息…')
            self._job(lambda: (path, engine.probe(path)), 'probe')

    def analyze(self):
        if not self.available():
            return
        if not self.source:
            raise ValueError('请先选择歌曲')
        is_full = bool(self.full_song.get())
        start = 0.0 if is_full else float(self.start.get())
        if not is_full and not getattr(self, 'source_probed', False):
            self.source_duration = float(engine.probe(self.source))
            self.source_probed = True
        duration = float(self.source_duration or self.duration.get() or 0.0) if is_full else float(self.duration.get())
        if not 0 <= start or (not is_full and (not 2 <= duration <= 3600 or start + duration > self.source_duration + 0.05)):
            raise ValueError('请选择歌曲范围内的 2 秒至 60 分钟音频')
        if not self.confirm_replace():
            return
        self.stop()
        self.progress['value'] = 0
        self.status.set('开始处理…')
        backend = 'rosvot' if self.model_choice.get().strip().startswith('演唱音符模型') else 'pyin'
        vocal_only = self.vocal_only.get()
        def _run_analysis():
            eff_dur = engine.probe(self.source) if is_full else duration
            if not 2 <= eff_dur <= 3600:
                raise ValueError('请选择歌曲范围内的 2 秒至 60 分钟音频')
            return engine.analyze(
                self.source, start, eff_dur, self.cancel,
                lambda p, s: self.events.put(('progress', (p, s))), backend, vocal_only
            )
        self._job(_run_analysis, 'result')

    def cancel_job(self):
        self.cancel.set()
        self.cancel_button.configure(state='disabled')
        self.status.set('正在取消，音高分析会在当前小段结束后停止…')

    def settings(self, project):
        project.bpm = float(self.bpm.get())
        project.meter = self.meter.get()
        project.tonic = self.tonic.get()
        project.beat_origin = float(self.origin.get())
        project.validate()
        return project

    def remember(self):
        self.stop()
        snap = copy.deepcopy(self.project)
        snap._saved_timing_choice = self.timing_choice.get()
        snap._saved_source_state = (self.source, self.source_duration, getattr(self, 'source_probed', False), self.clip_end, self.filename.get(), self.start.get(), self.duration.get())
        self.undo_stack = (self.undo_stack + [snap])[-30:]
        self.dirty = True

    def apply_settings(self):
        if not self.available():
            return
        candidate = self.settings(copy.deepcopy(self.project))
        self.remember()
        self.project = candidate
        self.refresh()
        self.status.set('设置已应用。音符拍数保持不变；如需按原录音重算节奏，请点「按原始时间重排」。')

    def requantize(self):
        if not self.available():
            return
        candidate = self.settings(copy.deepcopy(self.project))
        if not candidate.notes:
            raise ValueError('请先输入或提取旋律')
        if not messagebox.askyesno('重排节奏', '会按保留的原始时间重新量化，并覆盖手动修改的起始拍和时值；可以撤销。继续？', parent=self):
            return
        candidate.notes = quantize(candidate.notes, candidate.bpm, candidate.beat_origin)
        candidate.validate()
        self.remember(); self.project = candidate
        self.timing_choice.set('谱面节奏')
        self.refresh()
        self.status.set('已按原始时间重排，请检查切分音和三连音。')

    def from_manual(self):
        if not self.available():
            return
        candidate = self.settings(Project(title='手动输入旋律'))
        notes, beats = parse_jianpu(self.manual_text.get('1.0', 'end'), candidate.tonic, candidate.bpm, candidate.beat_origin)
        candidate.notes = notes
        candidate.clip_duration = candidate.beat_origin + beats * 60 / candidate.bpm
        candidate.validate()
        if not self.confirm_replace():
            return
        self.remember(); self.set_project(candidate, keep_undo=True)
        self.dirty = True
        self.status.set('手动旋律已生成，可试听、编辑并导出。')

    def set_project(self, project, keep_undo=False):
        self.stop()
        if not project.reference_notes:
            project.reference_notes = copy.deepcopy(project.notes)
        self.project = project
        self.timing_choice.set('谱面节奏' if project.reference_notes != project.notes else '原演唱节奏')
        if not keep_undo: self.undo_stack = []
        self.dirty = False
        self.bpm.set(f'{project.bpm:.2f}')
        self.meter.set(project.meter)
        self.tonic.set(project.tonic)
        self.origin.set(f'{project.beat_origin:.4f}')
        if project.source:
            old_source = self.source
            self.source = project.source
            self.clip_end = project.clip_start + project.clip_duration
            if old_source == project.source and getattr(self, 'source_probed', False):
                self.source_duration = max(self.clip_end, self.source_duration)
            elif Path(project.source).is_file():
                try:
                    self.source_duration = float(engine.probe(project.source))
                    self.source_probed = True
                except Exception as exc:
                    self.source_duration = self.clip_end
                    self.source_probed = False
                    self.status.set(f'音频总时长读取失败（{exc}），暂按片段结束时间估算。')
            else:
                self.source_duration = self.clip_end
                self.source_probed = False
            self.filename.set(Path(project.source).name)
            self.start.set(f'{project.clip_start:g}')
            self.duration.set(f'{project.clip_duration:g}')
        else:
            self.source = ''
            self.source_duration = 0
            self.source_probed = False
            self.clip_end = 0
            self.filename.set('尚未选择歌曲')
        self.refresh()

    def refresh(self):
        self.table.delete(*self.table.get_children())
        for i, note in enumerate(self.project.notes):
            low_conf = note.confidence < CONFIDENCE_THRESHOLD and not note.edited
            state = '已手动修改' if note.edited else '建议检查' if low_conf else '待试听'
            self.table.insert('', 'end', iid=str(i), values=(i + 1, numbered(note.pitch, self.project.tonic),
                  pitch_name(note.pitch) + f' / {note.pitch}', f'{note.start:g}', f'{note.duration:g}',
                  f'{self.project.clip_start + note.raw_start:.2f}', state),
                  tags=('uncertain',) if low_conf else ())
        count = sum(n.confidence < CONFIDENCE_THRESHOLD and not n.edited for n in self.project.notes)
        self.stats.set(f'{len(self.project.notes)} 个音符 · {count} 处建议检查 · 拖动音符移动，拖边缘改时值 · Ctrl 多选，空白处框选 · 修改可撤销')
        self.draw_roll()

    def draw_roll(self, elapsed=None):
        self.roll.delete('all')
        width = max(self.roll.winfo_width(), 500) * int(self.zoom.get())
        self.roll.configure(scrollregion=(0,0,width,self.roll.winfo_height()))
        height = self.roll.winfo_height()
        self.roll_hits = []
        if not self.project.notes:
            self.roll.create_text(width / 2, height / 2, text='旋律会显示在这里\n横向是时间，纵向是音高', fill=MUTED, font=('Microsoft YaHei UI', 12), justify='center')
            return
        notes = self.project.notes
        total = max(n.start + n.duration for n in notes) + 0.5
        low, high = min(n.pitch for n in notes) - 2, max(n.pitch for n in notes) + 2
        scale = (width - 68) / total
        ys = (height - 32) / (high - low + 1)
        self.roll_scale, self.roll_ys = scale, ys
        measure = int(self.project.meter.split('/')[0]) * 4 / int(self.project.meter.split('/')[1])
        import math
        bar_count = math.ceil(total / measure)
        stride = max(1, math.ceil(bar_count / 40))
        for bar in range(0, bar_count + 1, stride):
            x = 48 + bar * measure * scale
            self.roll.create_line(x, 22, x, height, fill='#2c3a4d')
            self.roll.create_text(x + 4, 10, text=str(bar + 1), fill=MUTED, anchor='w', font=('Segoe UI', 9))
        for pitch in range(low, high + 1):
            y = height - 10 - (pitch - low + 0.5) * ys
            if pitch % 12 == 0:
                self.roll.create_line(38, y, width, y, fill='#29384a')
                self.roll.create_text(5, y, text=pitch_name(pitch), fill=MUTED, anchor='w', font=('Segoe UI', 9))
        selected = set(self.table.selection()) if hasattr(self, 'table') else set()
        for i, note in enumerate(notes):
            x = 48 + note.start * scale
            y = height - 10 - (note.pitch - low + 0.5) * ys
            end = max(x + 3, x + note.duration * scale - 2)
            half = max(3, min(8, ys * 0.4))
            color = AMBER if note.confidence < CONFIDENCE_THRESHOLD and not note.edited else ACCENT
            self.roll.create_rectangle(x, y - half, end, y + half, fill=color, outline=FG if str(i) in selected else color)
            self.roll_hits.append((x, y - half - 2, end, y + half + 2, i))
        if elapsed is not None:
            beat = elapsed * self.project.bpm / 60
            if self.play_kind == 'raw':
                # Interpolate score beat position linearly within the active raw note span.
                before = [n for n in notes if n.raw_start <= elapsed]
                if before:
                    n = max(before, key=lambda item: (item.raw_start, item.start))
                    beat = n.start + min(n.duration, max(0, (elapsed - n.raw_start) / max(n.raw_duration, 0.001)) * n.duration)
                else:
                    beat = 0
            x = 48 + beat * scale
            self.roll.create_line(x, 20, x, height, fill='#ffffff', width=2)

    def pick_roll(self, event):
        if not self.available():
            return
        self.stop()
        x = self.roll.canvasx(event.x)
        for x1, y1, x2, y2, index in getattr(self, 'roll_hits', []):
            if x1 <= x <= x2 and y1 <= event.y <= y2:
                if event.state & 4:
                    if str(index) in self.table.selection():
                        self.table.selection_remove(str(index))
                    else:
                        self.table.selection_add(str(index))
                    return
                self.table.selection_set(str(index))
                self.table.see(str(index))
                mode = 'end' if abs(x-x2)<6 else 'start' if abs(x-x1)<6 else 'move'
                self.drag = (index,x,event.y,mode,copy.deepcopy(self.project.notes),self.roll_scale,self.roll_ys)
                return
        self.table.selection_remove(*self.table.selection())
        self.drag = (None,x,event.y,'box',None,0,0)

    def drag_roll(self, event):
        if not self.drag:
            return
        index,x,y,mode,notes,scale,ys=self.drag
        self.roll.delete('drag-preview')
        if mode=='box':
            self.roll.create_rectangle(x,y,self.roll.canvasx(event.x),event.y,outline=FG,tags='drag-preview')
        else:
            self.roll.create_line(self.roll.canvasx(event.x),20,self.roll.canvasx(event.x),self.roll.winfo_height(),fill=FG,tags='drag-preview')

    def release_roll(self, event):
        if not self.drag:
            return
        index,x,y,mode,notes,scale,ys=self.drag
        self.drag=None
        self.roll.delete('drag-preview')
        end=self.roll.canvasx(event.x)
        if mode=='box':
            selected=[str(i) for a,b,c,d,i in self.roll_hits if c>=min(x,end) and a<=max(x,end) and d>=min(y,event.y) and b<=max(y,event.y)]
            self.table.selection_set(*selected)
            return
        if abs(end-x)<3 and abs(event.y-y)<3:
            return
        delta=round((end-x)/scale*4)/4
        note=notes[index]
        if mode=='move':
            note.start=max(0,note.start+delta)
            note.pitch=max(21,min(108,note.pitch+round((y-event.y)/ys)))
        elif mode=='end':
            note.duration=max(.25,note.duration+delta)
        else:
            finish=note.start+note.duration
            note.start=max(0,min(finish-.25,note.start+delta))
            note.duration=finish-note.start
        note.edited=True
        self.commit_notes(notes)

    def organize_score(self, local=False):
        if not self.available():
            return
        from score_cleanup import organize
        indices=self.selected(False) if local else list(range(len(self.project.notes)))
        if not indices:
            raise ValueError('请先提取或输入旋律')
        if indices!=list(range(indices[0],indices[-1]+1)):
            raise ValueError('局部整理请选择连续音符')
        mode={'自动节奏':'auto','普通节奏':'binary','三连音':'triplet'}.get(self.rhythm_mode.get(),'auto')
        candidate=self.settings(copy.deepcopy(self.project))
        if not candidate.reference_notes:
            candidate.reference_notes=copy.deepcopy(candidate.notes)
        arranged=organize([candidate.notes[i] for i in indices],candidate.bpm,candidate.beat_origin,mode)
        candidate.notes[indices[0]:indices[-1]+1]=arranged
        candidate.validate()
        self.remember(); self.project=candidate; self.refresh()
        self.timing_choice.set('谱面节奏')
        self.status.set('谱面节奏已整理，识别参考保留。请试听；速度和拍点错误仍需先纠正。可撤销。')

    def restore_reference(self):
        if not self.available():
            return
        if not self.project.reference_notes:
            raise ValueError('没有保留的识别参考')
        self.commit_notes(copy.deepcopy(self.project.reference_notes))
        self.timing_choice.set('原演唱节奏')

    def collapse_selected(self):
        if not self.available():
            return
        indices=self.selected(False)
        if len(indices)<2 or indices!=list(range(indices[0],indices[-1]+1)):
            raise ValueError('请选择至少两个连续音符；归并使用音高输入框的 MIDI 音高')
        notes=copy.deepcopy(self.project.notes)
        selected=[notes[i] for i in indices]
        first,last=notes[indices[0]],notes[indices[-1]]
        raw_lo=min(n.raw_start for n in selected)
        raw_hi=max(n.raw_start+n.raw_duration for n in selected)
        first.pitch=int(self.pitch.get())
        first.duration=last.start+last.duration-first.start
        first.raw_start=raw_lo
        first.raw_duration=raw_hi-raw_lo
        first.edited=True
        notes[indices[0]:indices[-1]+1]=[first]
        self.commit_notes(notes)

    def loop_selected(self):
        if not self.available():
            return
        indices=self.selected(False)
        self.stop()
        project=copy.deepcopy(self.project)
        project.notes=[project.notes[i] for i in indices]
        offset=min(n.start for n in project.notes)
        for n in project.notes:
            n.start-=offset
        self.play_offset=offset*60/project.bpm
        job_id=getattr(self,'job_token',0)+1
        def prepare():
            engine.JOBS.mkdir(parents=True,exist_ok=True)
            path=engine.JOBS/f'preview-loop-{job_id}.wav'
            engine.synthesize(project,path,tail=0)
            return path,engine.probe(path),'loop'
        self._job(prepare,'play')

    def select_note(self, event=None):
        selected = self.table.selection()
        if selected:
            note = self.project.notes[int(selected[0])]
            self.pitch.set(str(note.pitch))
            self.note_start.set(f'{note.start:g}')
            self.note_duration.set(f'{note.duration:g}')
        self.draw_roll()

    def selected(self, single=True):
        selected = sorted(int(i) for i in self.table.selection())
        if not selected or (single and len(selected) != 1):
            raise ValueError('请先选择一个音符' if single else '请先选择音符')
        return selected

    def commit_notes(self, notes):
        candidate = copy.deepcopy(self.project)
        candidate.notes = notes
        candidate.validate()
        self.remember()
        self.project = candidate
        self.timing_choice.set('谱面节奏')
        self.refresh()
        self.status.set('修改已应用，可撤销。起始拍和时值影响乐谱试听；原始录音时间保留用于对齐。')

    def edit_note(self):
        if not self.available():
            return
        index = self.selected()[0]
        notes = copy.deepcopy(self.project.notes)
        note = notes[index]
        note.pitch, note.start, note.duration = int(self.pitch.get()), float(self.note_start.get()), float(self.note_duration.get())
        note.edited = True
        self.commit_notes(notes)

    def add_note(self):
        if not self.available():
            return
        pitch, start, duration = int(self.pitch.get()), float(self.note_start.get()), float(self.note_duration.get())
        raw_start = max(0, self.project.beat_origin + start * 60 / self.project.bpm)
        note = Note(pitch, start, duration, raw_start, duration * 60 / self.project.bpm, 1, True)
        self.commit_notes(copy.deepcopy(self.project.notes) + [note])

    def delete_notes(self):
        if not self.available():
            return
        selected = self.selected(False)
        self.commit_notes([copy.deepcopy(n) for i, n in enumerate(self.project.notes) if i not in selected])

    def split_note(self):
        if not self.available():
            return
        index = self.selected()[0]
        notes = copy.deepcopy(self.project.notes)
        original = notes.pop(index)
        if original.duration < 0.25:
            raise ValueError('此音符太短，无法继续等分')
        left = copy.deepcopy(original)
        left.duration /= 2
        left.raw_duration /= 2
        left.edited = True
        right = copy.deepcopy(left)
        right.start += left.duration
        right.raw_start += left.raw_duration
        notes[index:index] = [left, right]
        self.commit_notes(notes)

    def merge_notes(self):
        if not self.available():
            return
        indices = self.selected(False)
        if len(indices) < 2 or indices != list(range(indices[0], indices[-1] + 1)):
            raise ValueError('请按住 Ctrl 选择至少两个相邻音符')
        selected = [self.project.notes[i] for i in indices]
        if len({n.pitch for n in selected}) != 1:
            raise ValueError('合并仅支持相同音高，避免无意改变旋律')
        raw_lo = min(n.raw_start for n in selected)
        raw_hi = max(n.raw_start + n.raw_duration for n in selected)
        first = copy.deepcopy(selected[0])
        first.duration = selected[-1].start + selected[-1].duration - first.start
        first.raw_start = raw_lo
        first.raw_duration = raw_hi - raw_lo
        first.edited = True
        notes = copy.deepcopy(self.project.notes)
        notes[indices[0] : indices[-1] + 1] = [first]
        self.commit_notes(notes)

    def undo(self):
        if not self.available() or not self.undo_stack:
            return
        self.stop()
        saved = self.undo_stack.pop()
        if isinstance(saved, tuple):
            self.project, saved_timing = saved
        else:
            self.project = saved
            saved_timing = getattr(self.project, '_saved_timing_choice',
                '谱面节奏' if self.project.reference_notes != self.project.notes else '原演唱节奏')
        src_state = getattr(self.project, '_saved_source_state', None)
        if src_state is not None:
            self.source, self.source_duration, self.source_probed, self.clip_end, fn, st, du = src_state
            self.filename.set(fn); self.start.set(st); self.duration.set(du)
        self.timing_choice.set(saved_timing)
        self.bpm.set(f'{self.project.bpm:.2f}')
        self.meter.set(self.project.meter)
        self.tonic.set(self.project.tonic)
        self.origin.set(f'{self.project.beat_origin:.4f}')
        self.dirty = True; self.refresh()
        self.status.set('已撤销上一步修改。')

    def stop(self):
        self.play_token = getattr(self, 'play_token', 0) + 1
        self.play_pending = False
        if getattr(self, 'active_job_kind', None) == 'play':
            self.job_token = getattr(self, 'job_token', 0) + 1
            self.busy = False
            self.active_job_kind = None
            if hasattr(self, 'extract_button'):
                self.extract_button.configure(state='normal')
            if hasattr(self, 'choose_button'):
                self.choose_button.configure(state='normal')
            if hasattr(self, 'cancel_button'):
                self.cancel_button.configure(state='disabled')
        self.play_started = None
        self.looping = False
        self.play_offset = 0
        try:
            winsound.PlaySound(None, 0)
        except Exception as exc:
            self.status.set('停止播放异常：' + str(exc))
        if hasattr(self, 'roll'):
            self.draw_roll()

    def play(self, mode):
        if not self.available():
            return
        if not self.project.notes:
            raise ValueError('请先提取或输入旋律')
        self.stop()
        project = copy.deepcopy(self.project)
        raw_timing = self.timing_choice.get() == '原演唱节奏' and project.engine != 'manual'
        self.status.set('准备试听…')
        job_id = getattr(self, 'job_token', 0) + 1
        def prepare():
            engine.JOBS.mkdir(parents=True, exist_ok=True)
            if mode in ('mix', 'vocals'):
                path = Path(project.audio.get(mode, ''))
                if not path.is_file():
                    raise ValueError('此项目没有对应音轨；手动简谱可使用「旋律」试听')
                return path, engine.probe(path), 'raw'
            path = engine.JOBS / f'preview-{mode}-{job_id}.wav'
            if mode == 'compare':
                engine.compare_audio(project, path)
                return path, engine.probe(path), 'raw'
            engine.synthesize(project, path, raw=raw_timing)
            return path, engine.probe(path), 'raw' if raw_timing else 'grid'
        self._job(prepare, 'play')

    def save(self):
        if not self.available():
            return
        if not self.project.notes:
            raise ValueError('请先提取或输入旋律')
        path = filedialog.asksaveasfilename(title='保存旋律项目', defaultextension='.json', filetypes=[('旋律项目', '*.json')], initialfile=self.project.title + '.json')
        if path:
            self.project.save(path)
            self.dirty = False
            self.status.set('项目已保存。包含原始时间和校对结果；音频仍引用本机文件。')

    def guitar_dialog(self):
        if not self.available():
            return
        if not self.project.notes:
            raise ValueError('请先打开或提取旋律，建议完成谱面节奏整理后再编配')
        import guitar
        import json
        import uuid
        project=self.settings(copy.deepcopy(self.project))
        window=tk.Toplevel(self)
        window.title('吉他弹唱伴奏 · 编配草稿')
        window.geometry('760x520')
        window.configure(bg=BG)
        window.transient(self);window.grab_set()
        frame=ttk.Frame(window,padding=20);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='先确认调性，再修改每小节和弦',style='Title.TLabel').pack(anchor='w')
        candidates=guitar.key_candidates(project)
        labels=[name+suffix for name in NAMES for suffix in (' 大调',' 小调')]
        default_key=(NAMES[candidates[0][1]]+(' 小调' if candidates[0][2] else ' 大调')) if candidates else f'{project.tonic} 大调'
        key=tk.StringVar(value=default_key)
        style_options=['分解和弦','流行扫弦']
        capo_options=[str(i) for i in range(13)]
        style=tk.StringVar(value=style_options[0])
        capo=tk.StringVar(value='0')
        ttk.Label(frame,text='旋律调性候选：'+' / '.join(NAMES[r]+(' 小调' if m else ' 大调') for _,r,m in candidates)+'（可能混淆关系大小调）',style='Muted.TLabel').pack(anchor='w',pady=10)
        row=ttk.Frame(frame);row.pack(fill='x')
        for label,var,values,width in [('调性',key,labels,10),('风格',style,style_options,12),('变调夹',capo,capo_options,4)]:
            ttk.Label(row,text=label).pack(side='left',padx=(0,5))
            ttk.Combobox(row,textvariable=var,values=values,state='readonly',width=width).pack(side='left',padx=(0,14))
        ttk.Label(frame,text='每项对应一小节，可改为 C、Am、F#m、Bb 等；N.C. 表示不伴奏。\n和弦名称为实际音高，品位相对变调夹。仅凭旋律生成的候选不代表原曲和弦。',style='Muted.TLabel').pack(anchor='w',pady=12)
        chords=tk.Text(frame,height=9,bg=PANEL,fg=FG,insertbackground=FG,font=('Consolas',12),wrap='word')
        chords.pack(fill='both',expand=True)
        def fill(names):
            chords.delete('1.0','end')
            chords.insert('1.0','\n'.join('  '.join(f'{i+1}:{name}' for i,name in enumerate(names) if i//6==row) for row in range((len(names)+5)//6)))
        def suggest():
            root=NAMES.index(key.get().split()[0]);minor=key.get().endswith('小调')
            names=guitar.suggest(project,root,minor)
            fill(names)
            capo.set(str(guitar.suggest_capo(names)))
        def load_arrangement():
            path=filedialog.askopenfilename(parent=window,title='打开编配数据',filetypes=[('编配数据','*.json')])
            if not path:return
            if Path(path).stat().st_size>Project.MAX_PROJECT_BYTES:
                raise ValueError('编配文件过大')
            data=json.loads(Path(path).read_text(encoding='utf-8'))
            names=[bar['chord'] for bar in data['bars']]
            loaded_style=str(data['style'])
            loaded_capo=int(data['capo'])
            if loaded_style not in style_options or str(loaded_capo) not in capo_options:
                raise ValueError('不支持的编配风格或变调夹品位')
            guitar.arrange(project,names,loaded_style,loaded_capo)
            fill(names);style.set(loaded_style);capo.set(str(loaded_capo))
        def generate():
            names=[token.split(':')[-1] for token in chords.get('1.0','end').split()]
            chosen_style=style.get();chosen_capo=int(capo.get())
            if chosen_style not in style_options or str(chosen_capo) not in capo_options:
                raise ValueError('不支持的编配风格或变调夹品位')
            guitar.arrange(project,names,chosen_style,chosen_capo)
            directory=engine.ROOT.parent/'吉他编配'/uuid.uuid4().hex
            window.destroy()
            self.status.set('生成吉他六线谱、双轨 MIDI 和试听…')
            self._job(lambda:guitar.export_bundle(project,names,directory,chosen_style,chosen_capo),'guitar')
        row=ttk.Frame(frame);row.pack(fill='x',pady=(14,0))
        ttk.Button(row,text='重新建议和弦',command=suggest).pack(side='left')
        ttk.Button(row,text='打开编配数据',command=load_arrangement).pack(side='left',padx=8)
        ttk.Button(row,text='生成六线谱与试听',style='Accent.TButton',command=generate).pack(side='right')
        suggest()

    def lyrics_dialog(self):
        if not self.available():return
        import lyrics
        window=tk.Toplevel(self);window.title('歌词识别与校对');window.geometry('820x540')
        window.configure(bg=BG);window.transient(self);window.grab_set()
        frame=ttk.Frame(window,padding=18);frame.pack(fill='both',expand=True)
        ttk.Label(frame,text='每行：开始秒数 → Tab → 结束秒数 → Tab → 歌词。时间相对本次音频起点。\n自动歌词需校对；谱面按句子时间定位，不宣称逐字对应音符。',style='Muted.TLabel').pack(anchor='w')
        text=tk.Text(frame,bg=PANEL,fg=FG,insertbackground=FG,wrap='word',font=('Microsoft YaHei UI',11))
        text.pack(fill='both',expand=True,pady=10)
        text.insert('1.0',lyrics.lyric_text(self.project.lyrics))
        def save_lines():
            lines=lyrics.parse_text(text.get('1.0','end'),self.project.clip_duration)
            candidate=copy.deepcopy(self.project);candidate.lyrics=lines;candidate.validate()
            self.remember();self.project=candidate;window.destroy()
            self.status.set('歌词修改已保存到当前项目；请保存项目文件。')
        def recognize():
            path=Path(self.project.audio.get('vocals',''))
            if not path.is_file():raise ValueError('请先提取人声，或打开带有人声音轨的项目')
            window.destroy()
            self._job(lambda:lyrics.recognize(path,lambda:engine.check_cancel(self.cancel),lambda p,s:self.events.put(('progress',(p,s)))),'lyrics')
        row=ttk.Frame(frame);row.pack(fill='x')
        ttk.Button(row,text='从人声重新识别（中文）',command=recognize).pack(side='left')
        ttk.Button(row,text='保存校对',command=save_lines).pack(side='right')

    def open_project(self):
        if not self.available() or not self.confirm_replace():
            return
        path = filedialog.askopenfilename(title='打开旋律项目', filetypes=[('旋律项目', '*.json')])
        if path:
            self.set_project(Project.load(path))
            self.status.set('项目已打开；缺失的本机音轨不会影响旋律编辑和 MIDI 导出。')

    def export(self, kind):
        if not self.available():
            return
        if not self.project.notes:
            raise ValueError('请先提取或输入旋律')
        ext = '.mid' if kind == 'midi' else '.txt'
        path = filedialog.asksaveasfilename(title='导出旋律', defaultextension=ext, initialfile=self.project.title + ext,
                                          filetypes=[('MIDI' if kind == 'midi' else '简谱文本', '*' + ext)])
        if path:
            if kind == 'midi':
                Path(path).write_bytes(midi_bytes(self.project))
            else:
                Path(path).write_text(text_score(self.project), encoding='utf-8-sig')
            self.status.set('导出完成：' + Path(path).name)

    def close(self):
        if self.busy and not messagebox.askyesno('结束处理', '当前仍在处理音频。关闭将取消当前任务，是否继续？', parent=self):
            return
        if not self.confirm_replace():
            return
        self.cancel.set()
        self.stop()
        if self.busy:
            self.pending_close = True
            self.status.set('正在停止后台处理，完成后窗口会自动关闭…')
            return
        self.destroy()


if __name__ == '__main__':
    App().mainloop()
