using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

class Launcher {
    [STAThread]
    static void Main() {
        try {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string workspace = Directory.GetParent(root.TrimEnd(Path.DirectorySeparatorChar)).Parent.FullName;
            string python = Path.Combine(workspace, "work", "runtime", "venv", "Scripts", "pythonw.exe");
            if (!File.Exists(python)) {
                MessageBox.Show("运行环境尚未安装。请先按 README.md 完成安装。", "MelodyDesk");
                return;
            }
            var start = new ProcessStartInfo(python, "\"" + Path.Combine(root, "app.py") + "\"");
            start.WorkingDirectory = workspace;
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            Process.Start(start);
        } catch (Exception e) {
            MessageBox.Show(e.Message, "MelodyDesk 启动失败");
        }
    }
}
