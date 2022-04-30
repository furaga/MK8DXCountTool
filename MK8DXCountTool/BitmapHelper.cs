using System.Drawing;


namespace MK8DXCountTool
{
    class BitmapHelper
    {
        public static Bitmap LoadImage(string path)
        {
            Bitmap img = null;
            if (System.IO.File.Exists(path))
            {
                using (var bmp = Bitmap.FromFile(path))
                {
                    img = new Bitmap(bmp);
                }
            }
            return img;
        }

        public static Bitmap LoadAsThumbnail(string path, int w, int h)
        {
            Bitmap raw = LoadImage(path);
            Bitmap img = new Bitmap(w, h, raw.PixelFormat);
            using (var g = Graphics.FromImage(img))
            {
                float rx = (float)w / raw.Width;
                float ry = (float)h / raw.Height;
                float ratio = System.Math.Min(rx, ry);
                float iw = raw.Width * ratio;
                float ih = raw.Height * ratio;
                float ox = (img.Width - iw) / 2;
                float oy = (img.Height - ih) / 2;
                g.DrawImage(raw, ox, oy, iw, ih);
            }
            return img;
        }
    }
}