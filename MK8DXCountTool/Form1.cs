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

            updateList();
        }

        void updateList()
        {
            listView.Clear();
            foreach(var name in imageNames)
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
            }
            if (e.Button == MouseButtons.Right)
            {
                countDict_[name] = Math.Max(0, countDict_[name] - 1);
                item.Text = "Count: " + countDict_[name];
            }
        }

        void saveCountDict()
        {

        }
    }
}
