using UnityEngine;
using System.Drawing;
using System.Windows.Forms;
using System.Threading;

/*
This script creates a new thread that runs a loop which takes screenshots every
interval seconds and saves them to the specified savePath directory.

To use the script, simply attach it to a GameObject in your Unity scene.

The Start() method will be called once the script is loaded, and it will start
the screenshot loop in a new thread. The TakeScreenshots() function will be
called repeatedly in the background, taking screenshots every interval seconds
and saving them to the savePath directory.
*/

public class Screenshotter : MonoBehaviour
{
    public int interval = 10; // interval in seconds
    public string savePath = @"C:\Screenshots\"; // path to save screenshots
    private int screenshotCounter = 1;

    // Start is called before the first frame update
    void Start()
    {
        // start the screenshot loop in a new thread
        Thread screenshotThread = new Thread(new ThreadStart(TakeScreenshots));
        screenshotThread.Start();
    }

    // function to take screenshots every 'interval' seconds
    void TakeScreenshots()
    {
        // infinite loop to take screenshots every 'interval' seconds
        while (true)
        {
            // create a bitmap object to store the screenshot
            Bitmap screenshot = new Bitmap(Screen.PrimaryScreen.Bounds.Width, Screen.PrimaryScreen.Bounds.Height);

            // create a graphics object to draw the screenshot
            Graphics graphics = Graphics.FromImage(screenshot);

            // copy the contents of the primary display to the bitmap
            graphics.CopyFromScreen(Screen.PrimaryScreen.Bounds.X, Screen.PrimaryScreen.Bounds.Y, 0, 0, Screen.PrimaryScreen.Bounds.Size);

            // save the screenshot to a file with a timestamped filename
            string filename = $"Screenshot_{screenshotCounter}_{System.DateTime.Now:yyyyMMdd_HHmmss}.jpg";
            screenshot.Save(savePath + filename);

            // increment the screenshot counter
            screenshotCounter++;

            // wait for 'interval' seconds before taking the next screenshot
            Thread.Sleep(interval * 1000);
        }
    }
}
