using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

class Launcher {
    static string FindPythonw(string root) {
        string[] relCandidates = new string[] {
            Path.Combine(root, ".venv", "Scripts", "pythonw.exe"),
            Path.Combine(root, "work", "runtime", "venv", "Scripts", "pythonw.exe")
        };
        foreach (string c in relCandidates) {
            if (File.Exists(c)) return c;
        }
        DirectoryInfo d1 = Directory.GetParent(root.TrimEnd(Path.DirectorySeparatorChar));
        DirectoryInfo d2 = d1 != null ? d1.Parent : null;
        if (d2 != null) {
            string wsPython = Path.Combine(d2.FullName, "work", "runtime", "venv", "Scripts", "pythonw.exe");
            if (File.Exists(wsPython)) return wsPython;
        }
        return null;
    }

    [STAThread]
    static void Main() {
        try {
            string root = AppDomain.CurrentDomain.BaseDirectory;
            string python = FindPythonw(root);
            if (python == null) {
                MessageBox.Show("本地开发虚拟环境尚未安装。免安装用户请直接下载 GitHub Releases 中的便携版压缩包。", "MelodyDesk 2.0");
                return;
            }
            var start = new ProcessStartInfo(python, "\"" + Path.Combine(root, "melodydesk_desktop.py") + "\"");
            start.WorkingDirectory = root;
            start.UseShellExecute = false;
            start.CreateNoWindow = true;
            Process.Start(start);
        } catch (Exception e) {
            MessageBox.Show(e.Message, "MelodyDesk 启动失败");
        }
    }
}
