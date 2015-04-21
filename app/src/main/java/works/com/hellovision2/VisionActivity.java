package works.com.hellovision2;

import android.graphics.Color;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.TextView;

import com.androidplot.xy.BoundaryMode;
import com.androidplot.xy.LineAndPointFormatter;
import com.androidplot.xy.SimpleXYSeries;
import com.androidplot.xy.XYPlot;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.imgproc.Imgproc;

import com.badlogic.gdx.audio.analysis.FFT;

import java.util.Calendar;


public class VisionActivity extends ActionBarActivity implements CameraBridgeViewBase.CvCameraViewListener2{

    // Band pass filter configuration
    /*

FIR filter designed with
http://t-filter.appspot.com

sampling frequency: 15 Hz

* 0 Hz - 0.75 Hz
  gain = 0
  desired attenuation = -40 dB
  actual attenuation = -40.673652401924315 dB

* 1 Hz - 3.5 Hz
  gain = 1
  desired ripple = 5 dB
  actual ripple = 3.8535583688956008 dB

* 4 Hz - 7.5 Hz
  gain = 0
  desired attenuation = -40 dB
  actual attenuation = -40.673652401924315 dB

*/

    private static double filter_taps[] = {
            -0.01582872505016952,
            -0.021320649733125918,
            -0.004728041650791583,
            0.027258111294177816,
            0.04298479464374167,
            0.025125252743414244,
            -0.006197403698522914,
            -0.01873628079403876,
            -0.00849174701106352,
            -0.00120493566713456,
            -0.013389314669616478,
            -0.027534733875243366,
            -0.01923188790613837,
            0.001941799483073626,
            0.0053608341310428876,
            -0.007523699349047374,
            -0.0012917441210153746,
            0.029079799680659976,
            0.03979956098438757,
            0.014509871198566246,
            0.0027925469407884197,
            0.03133681998056229,
            0.04005920781156626,
            -0.014794889713598427,
            -0.06088522275024569,
            -0.027319476414450792,
            0.0008065846342549088,
            -0.09128039663456836,
            -0.2005839938734868,
            -0.09786601675499218,
            0.19882455158526804,
            0.36591259096164835,
            0.19882455158526804,
            -0.09786601675499218,
            -0.2005839938734868,
            -0.09128039663456836,
            0.0008065846342549053,
            -0.027319476414450795,
            -0.06088522275024569,
            -0.014794889713598434,
            0.04005920781156626,
            0.03133681998056229,
            0.0027925469407884215,
            0.014509871198566246,
            0.03979956098438757,
            0.02907979968065998,
            -0.0012917441210153675,
            -0.007523699349047374,
            0.0053608341310428945,
            0.001941799483073626,
            -0.01923188790613837,
            -0.027534733875243352,
            -0.013389314669616478,
            -0.00120493566713456,
            -0.008491747011063517,
            -0.018736280794038752,
            -0.006197403698522914,
            0.025125252743414244,
            0.04298479464374167,
            0.027258111294177816,
            -0.004728041650791583,
            -0.021320649733125918,
            -0.01582872505016952
    };

    private final int NUM_TAPS = 63;

    private static final int HISTORY_SIZE = 500;

    private static final int FFT_SIZE = 256;


    String TAG = "APP";
    CameraBridgeViewBase mOpenCvCameraView;

    private XYPlot cameraHistoryPlot = null;
    private XYPlot fftPlot = null;

    private SimpleXYSeries rawFftData = null;
    private SimpleXYSeries rawDataSeries = null;
    private SimpleXYSeries smoothedDataSeries = null;
    private SimpleXYSeries demeanedDataSeries = null;

    private TextView textviewBeats;


    int currentWindow = 0;
    float[] windowValues = new float[NUM_TAPS+FFT_SIZE];
    long windowStart = 0;
    long windowEnd = 0;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");
                    mOpenCvCameraView.enableView();
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    CircularProgressBar circularProgressbar;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_vision);

        circularProgressbar = (CircularProgressBar) findViewById(R.id.circularprogressbar1);
        circularProgressbar.setTitle("00");
        circularProgressbar.setSubTitle("Ready");
        circularProgressbar.setProgress(0);

        textviewBeats = (TextView) findViewById(R.id.textviewBeats);

        // setup the camera plot:
        cameraHistoryPlot = (XYPlot) findViewById(R.id.cameraHistoryPlot);

        rawDataSeries = new SimpleXYSeries("Raw Signal");
        rawDataSeries.useImplicitXVals();
        smoothedDataSeries = new SimpleXYSeries("Smoothed Signal");
        smoothedDataSeries.useImplicitXVals();
        demeanedDataSeries = new SimpleXYSeries("Demeaned Signal");
        demeanedDataSeries.useImplicitXVals();

        // Camera plot, y axis is intensity, goes upto 255 max, X axis is time
        cameraHistoryPlot.setRangeBoundaries(-10, 255, BoundaryMode.FIXED);
        cameraHistoryPlot.setDomainBoundaries(0, 500, BoundaryMode.FIXED);

        cameraHistoryPlot.addSeries(rawDataSeries, new LineAndPointFormatter(Color.rgb(100, 100, 200), null, null, null));
        cameraHistoryPlot.addSeries(smoothedDataSeries, new LineAndPointFormatter(Color.rgb(100, 200, 100), null,null, null));
        cameraHistoryPlot.addSeries(demeanedDataSeries, new LineAndPointFormatter(Color.rgb(200, 100, 100), null, null,null));
        cameraHistoryPlot.setDomainStepValue(5);
        cameraHistoryPlot.setTicksPerRangeLabel(3);
        cameraHistoryPlot.setDomainLabel("Intensity");
        cameraHistoryPlot.getDomainLabelWidget().pack();
        cameraHistoryPlot.setRangeLabel("Time)");
        cameraHistoryPlot.getRangeLabelWidget().pack();

        fftPlot = (XYPlot) findViewById(R.id.fftPlot);
        rawFftData = new SimpleXYSeries("FFT");
        rawFftData.useImplicitXVals();
        fftPlot.setRangeBoundaries(-10, 1000, BoundaryMode.FIXED);
        fftPlot.setDomainBoundaries(0, 128, BoundaryMode.FIXED);
        fftPlot.addSeries(rawFftData, new LineAndPointFormatter(Color.rgb(100,100,200),null,null,null));


        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        mOpenCvCameraView = (CameraBridgeViewBase) findViewById(R.id.HelloOpenCvView);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);

    }


    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_6, this, mLoaderCallback);
    }


    @Override
    public void onPause()
    {
        super.onPause();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null)
            mOpenCvCameraView.disableView();
    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        // Inflate the menu; this adds items to the action bar if it is present.
        getMenuInflater().inflate(R.menu.menu_vision, menu);
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        // Handle action bar item clicks here. The action bar will
        // automatically handle clicks on the Home/Up button, so long
        // as you specify a parent activity in AndroidManifest.xml.
        int id = item.getItemId();

        //noinspection SimplifiableIfStatement
        if (id == R.id.action_settings) {
            return true;
        }

        return super.onOptionsItemSelected(item);
    }

    @Override
    public void onCameraViewStarted(int width, int height) {

    }
//
    @Override
    public void onCameraViewStopped() {

    }

    public void btnStartClick(View view)
    {
        Button button = (Button)view;
        if(measuring)
        {
            circularProgressbar.setTitle("00");
            circularProgressbar.setSubTitle("Ready");
            circularProgressbar.setProgress(0);
            button.setText("Start");
            measuring = false;
        }
        else {
            button.setText("Cancel");
            measuring = true;
        }
    }

    private int beats = 0;
    private boolean measuring = false;

    // Function to apply band pass filter on input signal
    // Returns the filtered signal values
    private float[] applyBandPassFilter(float[] signal)
    {
        float [] output = new float[signal.length];
        for(int i=0; i<signal.length; i++)
        {
            int convolutionLimit = NUM_TAPS;

            if(convolutionLimit>i)
            {
                convolutionLimit = i;
            }
            int index = i;
            int convolution = 0;

            // Apply band pass filter by using convolution
            while(convolution < convolutionLimit)
            {
                output[i] += signal[index] * filter_taps[convolution];
                convolution++;
                index --;
            }
        }
        return output;
    }

    // Calculates FFT for an input signal
    private float[] calculateFFT(float [] signal, float frequency)
    {
        FFT fft1 = new FFT(FFT_SIZE,frequency);

        float [] localSignal = new float[FFT_SIZE];

        // Zero Pad signal
        for (int i = 0; i < FFT_SIZE; i++) {

            if (i < signal.length) {
                localSignal[i] = signal[i];
            } else {
                localSignal[i] = 0;
            }
        }

        fft1.forward(localSignal);

        float[] imag = fft1.getImaginaryPart();
        float[] real = fft1.getRealPart();

        float[] mag = new float[FFT_SIZE];

        for (int i = 0; i < FFT_SIZE; i++) {
            mag[i] = (float)Math.sqrt((real[i] * real[i]) + (imag[i] * imag[i]));
        }

        // Set this to 0 to ignore DC noise from FFT output
        mag[0] =0;
        return mag;
    }

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {

        Mat currentFrame = inputFrame.rgba();

        if(!measuring)
        {
            return currentFrame;
        }

        // check image size returned from camera
        int cols = currentFrame.cols();
        int rows = currentFrame.rows();

        // Pick the center of image, this is where we take pixel values to detect intensity
        int centerCol = cols/2;
        int centerRow = rows/2;

        int avgIntensity = 0;
        int sumOfAvgIntensity = 0;
        int pixelCount = 0;

        // Calculate average intensity over 40x40 area from center of the image
        for(int row=centerRow-20; row<centerRow+20; row++)
        {
            for(int col=centerCol-20; col<centerCol+20; col++)
            {
                double [] val = currentFrame.get(row,col);
                sumOfAvgIntensity+=((val[0]+val[1]+val[2])/3);
                pixelCount++;
            }
        }

        avgIntensity = sumOfAvgIntensity/pixelCount;

        // We have the average intensity value detected from this image
        // Now we add it to the raw data series

        // First make sure we make enough room in both raw data and filtered data series

        // We add one item at a time in raw data, so just delete last one
        if (rawDataSeries.size() > HISTORY_SIZE) {
            rawDataSeries.removeFirst();
        }

        // Filtered data is added in batches of (fft size + num taps) so we do batch delete here.
        while(smoothedDataSeries.size()>HISTORY_SIZE) {
            smoothedDataSeries.removeFirst();
        }

        // Now add the detected intensity to raw data.
        rawDataSeries.addLast(null, avgIntensity);

        // Redraw the camera plot to refresh the data
        cameraHistoryPlot.redraw();

        // Add the measure intensity to our window values array as well
        windowValues[currentWindow] = avgIntensity;


        // Update progress

        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                circularProgressbar.setTitle("00");
                circularProgressbar.setSubTitle("Detecting...");
                //circularProgressbar.setProgress(currentWindow);
                circularProgressbar.setProgress(((currentWindow*100)/(FFT_SIZE+NUM_TAPS)));
            }
        });


        // Window handling code that processes data in windows
        if(currentWindow == 0)
        {
            // We just started a new measuring window
            // Keep track of start time to calculate FPS.
            windowStart = System.currentTimeMillis();
        }

        currentWindow++;

        // The next block is executed every time we exceed the window size
        // This mean we have a complete window of data that needs to be processed
        // Window size is size of fft + num taps. This is because we ignore the first set of values
        // upto num taps since they are partial due to filtering.
        if(currentWindow>=NUM_TAPS+FFT_SIZE)
        {
            // Reset window back to 0
            currentWindow = 0;

            // Set the window end time for FPS calculation
            windowEnd = System.currentTimeMillis();
            long windowDurationInMillis = windowEnd - windowStart;

            // Calculate the FPS
            float cameraFrequency = (NUM_TAPS+FFT_SIZE) / (windowDurationInMillis / 1000);

            // Send the signal through band pass filter
            windowValues = applyBandPassFilter(windowValues);

            // add filtered values to chart
            // The 0th to num taps value is added only to smoothed data chart for display
            // We drop these values from FFT calculation to maintain accuracy
            for(int i = 0 ; i<NUM_TAPS; i++)
            {
                smoothedDataSeries.addLast(null,0);
            }

            // This var stores the accurate filtered values only of FFT size, so we dont add the intial values from filtering here.
            float [] filteredValues = new float[FFT_SIZE];
            for(int i=NUM_TAPS; i<NUM_TAPS+FFT_SIZE; i++)
            {
                smoothedDataSeries.addLast(null,windowValues[i]);
                filteredValues[i-NUM_TAPS] = windowValues[i];
            }

            // Now calculate the FFT using the filtered data that we obtained in last step
            float [] fft = calculateFFT(filteredValues, cameraFrequency);

            int highestFrequency = 0;
            float highestVal =0;

            // Check the first half of the FFT and pick the frequency value with highest amplitude
            // This should be the pulse rate frequency
            for (int i = 0; i < FFT_SIZE/2; i++) {

                if(i > 0 && fft[i]>highestVal)
                {
                    highestVal = fft[i];
                    highestFrequency = i;
                }
            }

            // Calculate the mapping off FFT index back to hz
            float hz = (float)(highestFrequency * (cameraFrequency/2)) / 128;

            // Calculate beats per second from the hertz value.
            beats = (int)(hz*60);

            // Update the UI to diplay the detected beats
            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    textviewBeats.setText(Integer.toString(beats));
                    circularProgressbar.setTitle(Integer.toString(beats));
                    circularProgressbar.setSubTitle("bpm");
                }
            });


            // Display the calculated FFT on the chart
            int size = rawFftData.size();
            for(int i=0; i<size; i++)
            {
                rawFftData.removeLast();
            }

            for(int i=0; i<fft.length; i++)
            {
                rawFftData.addLast(null, fft[i]);
            }

            fftPlot.redraw();
        }

        return currentFrame;
    }
}
