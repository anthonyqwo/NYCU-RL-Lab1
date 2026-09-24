"""Interactive 2048 using the trained C++ models, without changing weights."""
import json
import subprocess
from pathlib import Path
import tkinter as tk
from tkinter import ttk, messagebox
import argparse
import random

ROOT = Path(__file__).resolve().parents[1]
RUN = '20260924_202933'
DIRECTIONS = ['↑ 上', '→ 右', '↓ 下', '← 左']
COLORS = {0: '#cdc1b4', 2: '#eee4da', 4: '#ede0c8', 8: '#f2b179',
          16: '#f59563', 32: '#f67c5f', 64: '#f65e3b', 128: '#edcf72',
          256: '#edcc61', 512: '#edc850', 1024: '#edc53f', 2048: '#edc22e'}


class Engine:
    def __init__(self, variant):
        self.log = (ROOT / 'logs/gui_engine.log').open('a', encoding='utf-8')
        self.proc = subprocess.Popen(
            [str(ROOT / f'build/gui_{variant}.exe'), str(ROOT / f'models/{variant}_100k_{RUN}.bin')],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.log,
            text=True, bufsize=1, cwd=ROOT,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        self.initial = self.read()

    def read(self):
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError('模型程序已停止，請查看 logs/gui_engine.log。')
        return json.loads(line)

    def command(self, text):
        self.proc.stdin.write(text + '\n')
        self.proc.stdin.flush()
        return self.read()

    def close(self):
        if self.proc.poll() is None:
            try:
                self.proc.stdin.write('quit\n'); self.proc.stdin.flush()
                self.proc.wait(timeout=2)
            except (OSError, subprocess.TimeoutExpired):
                self.proc.kill(); self.proc.wait()
        self.proc.stdin.close(); self.proc.stdout.close(); self.log.close()


class Game:
    def __init__(self, root):
        self.root, self.engine, self.timer = root, None, None
        self.state = None
        self.running = False
        root.title('2048 · TD 模型測試 | 唐予安 315551017')
        root.configure(bg='#faf8ef')
        root.geometry('650x860')
        root.minsize(650, 860)
        root.protocol('WM_DELETE_WINDOW', self.close)
        style = ttk.Style()
        style.theme_use('clam')
        style.configure('TFrame', background='#faf8ef')
        style.configure('TLabel', background='#faf8ef', foreground='#574f47', font=('Microsoft JhengHei', 10))
        style.configure('TButton', font=('Microsoft JhengHei', 10), padding=7)
        outer = ttk.Frame(root, padding=20)
        outer.pack(fill='both', expand=True)
        title = ttk.Frame(outer); title.pack(fill='x')
        ttk.Label(title, text='2048', font=('Arial', 32, 'bold')).pack(side='left')
        self.score = ttk.Label(title, text='分數 0  |  步數 0', font=('Microsoft JhengHei', 14, 'bold'))
        self.score.pack(side='right', pady=15)
        ttk.Label(outer, text='已訓練 100,000 局的模型 · 僅推論，不更新權重').pack(anchor='w', pady=(0, 12))
        tools = ttk.Frame(outer); tools.pack(fill='x')
        self.variant = tk.StringVar(value='TD-state · 達成率 79.1%')
        self.selector = ttk.Combobox(tools, textvariable=self.variant, state='readonly', width=28,
                                    values=['TD-state · 達成率 79.1%', 'TD-after-state · 達成率 76.7%'])
        self.selector.pack(side='left')
        self.selector.bind('<<ComboboxSelected>>', lambda _: self.load())
        ttk.Label(tools, text='Seed').pack(side='left', padx=(12, 4))
        self.seed = tk.StringVar(value='2026')
        ttk.Entry(tools, textvariable=self.seed, width=10).pack(side='left')
        ttk.Button(tools, text='重開', command=self.reset).pack(side='right')
        self.canvas = tk.Canvas(outer, width=500, height=500, bg='#bbada0', highlightthickness=0)
        self.canvas.pack(pady=14)
        self.canvas.bind('<Button-1>', lambda _: self.canvas.focus_set())
        buttons = ttk.Frame(outer); buttons.pack(fill='x')
        self.auto = ttk.Button(buttons, text='▶ 自動遊玩', command=self.toggle)
        self.auto.pack(side='left')
        ttk.Button(buttons, text='AI 單步', command=self.step).pack(side='left', padx=7)
        ttk.Button(buttons, text='隨機新局', command=self.random_game).pack(side='left')
        ttk.Label(buttons, text='間隔 ms').pack(side='left', padx=(12, 4))
        self.speed = tk.IntVar(value=120)
        ttk.Spinbox(buttons, from_=20, to=1500, increment=20, textvariable=self.speed, width=6).pack(side='left')
        self.hint = ttk.Label(outer, text='載入模型中…', font=('Microsoft JhengHei', 11, 'bold'))
        self.hint.pack(anchor='w', pady=(13, 5))
        self.values = ttk.Label(outer, text='', font=('Microsoft JhengHei', 9))
        self.values.pack(anchor='w')
        ttk.Label(outer, text='手動：方向鍵／WASD　　空白鍵：播放／暫停\n取得 2048 後繼續遊玩，直到無合法動作。',
                  font=('Microsoft JhengHei', 9)).pack(anchor='w', pady=(8, 0))
        for key, action in [('Up', 0), ('Right', 1), ('Down', 2), ('Left', 3),
                            ('w', 0), ('d', 1), ('s', 2), ('a', 3)]:
            root.bind(key, lambda event, a=action: self.manual(event, a))
        root.bind('<space>', self.space)
        root.after(100, self.load)

    def stop(self):
        self.running = False
        self.auto.config(text='▶ 自動遊玩')
        if self.timer is not None:
            self.root.after_cancel(self.timer); self.timer = None

    def load(self):
        self.stop()
        if self.engine: self.engine.close(); self.engine = None
        try:
            self.hint.config(text='載入模型中…'); self.root.update_idletasks()
            variant = 'td_afterstate' if self.variant.get().startswith('TD-after') else 'td_state'
            self.engine = Engine(variant)
            self.reset()
        except Exception as exc:
            messagebox.showerror('無法載入模型', str(exc))

    def reset(self):
        self.stop()
        try:
            value = int(self.seed.get())
            if value < 0 or value > 4294967295: raise ValueError()
        except ValueError:
            messagebox.showerror('Seed 格式', 'Seed 必須是 0 到 4294967295 的整數。'); return
        if self.engine:
            self.request(f'reset {value}')
            self.canvas.focus_set()

    def random_game(self):
        self.seed.set(str(random.randrange(4294967296))); self.reset()

    def request(self, command):
        try:
            self.state = self.engine.command(command); self.draw()
        except Exception as exc:
            self.stop(); messagebox.showerror('遊戲錯誤', str(exc))

    def draw(self):
        s = self.state
        self.canvas.delete('all')
        for i, value in enumerate(s['board']):
            x, y = 12 + (i % 4) * 122, 12 + (i // 4) * 122
            self.canvas.create_rectangle(x, y, x+110, y+110, fill=COLORS.get(value, '#3c3a32'), outline='')
            if value:
                self.canvas.create_text(x+55, y+55, text=str(value),
                    fill='#776e65' if value <= 4 else '#fffaf0',
                    font=('Arial', 30 if value < 1024 else 24, 'bold'))
        self.score.config(text=f"分數 {s['score']:,}  |  步數 {s['moves']}")
        if s['terminal']:
            self.stop()
            self.hint.config(text=f"遊戲結束 · 最大方塊 {max(s['board'])} · {'達成 2048' if max(s['board']) >= 2048 else '未達 2048'}")
        else:
            reward = f"　上一步 +{s['reward']}" if s['moves'] and s['reward'] >= 0 else ''
            self.hint.config(text=f"AI 建議：{DIRECTIONS[s['best_action']]}{reward}")
        self.values.config(text='動作估值：' + '   '.join(
            f"{DIRECTIONS[a][0]} {'不可移動' if v is None else format(v, ',.0f')}" for a, v in enumerate(s['values'])))

    def step(self):
        self.stop()
        if self.engine and self.state and not self.state['terminal']: self.request('ai')

    def tick(self):
        self.timer = None
        if not self.running: return
        self.request('ai')
        if self.running:
            try: delay = max(20, min(1500, int(self.speed.get())))
            except (ValueError, tk.TclError): delay = 120
            self.timer = self.root.after(delay, self.tick)

    def toggle(self):
        if self.running: self.stop()
        elif self.engine and self.state and not self.state['terminal']:
            self.running = True; self.auto.config(text='Ⅱ 暫停'); self.tick()

    def manual(self, event, action):
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox, ttk.Spinbox)): return
        self.stop()
        if self.engine and self.state and not self.state['terminal']: self.request(f'move {action}')
        return 'break'

    def space(self, event):
        if isinstance(event.widget, (ttk.Entry, ttk.Combobox, ttk.Spinbox)): return
        self.toggle(); return 'break'

    def close(self):
        self.stop()
        if self.engine: self.engine.close()
        self.root.destroy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    root = tk.Tk()
    app = Game(root)
    if args.self_test:
        def check():
            assert app.engine and len(app.state['board']) == 16
            app.step(); assert app.state['moves'] == 1
            app.reset(); assert app.state['moves'] == 0
            app.variant.set('TD-after-state · 達成率 76.7%'); app.load()
            app.step(); assert app.state['moves'] == 1
            app.toggle(); assert app.running
            app.stop(); assert not app.running
            print('GUI self-test passed: load, AI step, reset, model switch, autoplay/pause.')
            app.close()
        root.after(1000, check)
    root.mainloop()
