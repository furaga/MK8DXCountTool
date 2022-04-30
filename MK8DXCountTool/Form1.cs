using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

namespace MK8DXCountTool
{
    public partial class Form1 : Form
    {
        const string ImageDir = @"data/images";
        List<string> imageNames = new List<string>();

        Dictionary<string, int> countDict_ = new Dictionary<string, int>();

        public Form1()
        {
            InitializeComponent();
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            reload();
        }

        void reload()
        {
            imageList.Images.Clear();
            imageNames.Clear();
            foreach (string imgPath in System.IO.Directory.GetFiles(ImageDir))
            {
                string filename = System.IO.Path.GetFileNameWithoutExtension(imgPath);
                var img = BitmapHelper.LoadAsThumbnail(imgPath, imageList.ImageSize.Width, imageList.ImageSize.Height);
                imageList.Images.Add(filename, img);
                imageNames.Add(filename);
                countDict_[filename] = 0;
            }

            imageNames.Sort();
            loadCountDict();

            updateList();
            updateStatistics();
        }

        void loadCountDict()
        {
            if (!System.IO.File.Exists("data/count.csv"))
            {
                return;
            }
            var lines = System.IO.File.ReadAllLines("data/count.csv");
            for (int i = 1; i < lines.Length; i++)
            {
                var tokens = lines[i].Trim().Split(',');
                string name = tokens[0];
                int count = int.Parse(tokens[1]);
                System.Diagnostics.Debug.Assert(countDict_.ContainsKey(name));
                countDict_[name] = count;
            }
        }

        void updateList()
        {
            listView.Clear();
            foreach (var name in imageNames)
            {
                listView.Items.Add("Count: " + countDict_[name], name);
            }
        }

        private void listView_MouseClick(object sender, MouseEventArgs e)
        {
            if (listView.SelectedItems.Count != 1)
            {
                return;
            }

            var item = listView.SelectedItems[0];
            string name = item.ImageKey;
            if (e.Button == MouseButtons.Left)
            {
                countDict_[name] += 1;
                item.Text = "Count: " + countDict_[name];
                saveCountDict();
                updateStatistics();
            }
            if (e.Button == MouseButtons.Right)
            {
                countDict_[name] = Math.Max(0, countDict_[name] - 1);
                item.Text = "Count: " + countDict_[name];
                saveCountDict();
                updateStatistics();
            }
        }

        void saveCountDict()
        {
            List<string> lines = new List<string>();
            lines.Add("Cource,Count");

            foreach (var kv in countDict_)
            {
                lines.Add(kv.Key + "," + kv.Value);
            }

            System.IO.File.WriteAllLines("data/count.csv", lines, Encoding.UTF8);
        }

        void updateStatistics()
        {
            int sum = 0;
            foreach (var kv in countDict_)
            {
                sum += kv.Value;
            }

            Dictionary<string, float> percentages = new Dictionary<string, float>();
            foreach (var kv in countDict_)
            {
                percentages[kv.Key] = 100.0f * kv.Value / sum;
            }

            List<string> names = percentages.Keys.ToList();
            names = names.OrderByDescending(k => countDict_[k]).ToList();

            richTextBox1.Clear();
            foreach (var name in names)
            {
                richTextBox1.Text += name + ": " + countDict_[name] + " (" + percentages[name] + "%)\n";
            }
        }

    }

}
