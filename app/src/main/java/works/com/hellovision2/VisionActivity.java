package works.com.hellovision2;

import android.graphics.Color;
import android.support.v7.app.ActionBarActivity;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.SurfaceView;
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

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        setContentView(R.layout.activity_vision);

        textviewBeats = (TextView) findViewById(R.id.textviewBeats);

        // setup the accelerometer plot:
        cameraHistoryPlot = (XYPlot) findViewById(R.id.cameraHistoryPlot);

        rawDataSeries = new SimpleXYSeries("Raw Signal");
        rawDataSeries.useImplicitXVals();
        smoothedDataSeries = new SimpleXYSeries("Smoothed Signal");
        smoothedDataSeries.useImplicitXVals();
        demeanedDataSeries = new SimpleXYSeries("Demeaned Signal");
        demeanedDataSeries.useImplicitXVals();

        cameraHistoryPlot.setRangeBoundaries(-10, 255, BoundaryMode.FIXED);
        cameraHistoryPlot.setDomainBoundaries(0, 500, BoundaryMode.FIXED);

        cameraHistoryPlot.addSeries(rawDataSeries, new LineAndPointFormatter(Color.rgb(100, 100, 200), null, null, null));
        cameraHistoryPlot.addSeries(smoothedDataSeries, new LineAndPointFormatter(Color.rgb(100, 200, 100), null,null, null));
        cameraHistoryPlot.addSeries(demeanedDataSeries, new LineAndPointFormatter(Color.rgb(200, 100, 100), null, null,null));
        cameraHistoryPlot.setDomainStepValue(5);
        cameraHistoryPlot.setTicksPerRangeLabel(3);
        cameraHistoryPlot.setDomainLabel("Acceleration");
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
//

    private int peak = 0;
    private int beats = 0;


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
            while(convolution < convolutionLimit)
            {
                output[i] += signal[index] * filter_taps[convolution];
                convolution++;
                index --;
            }

        }

        return output;
    }

    private float[] calculateFFT(float [] signal, float frequency)
    {
        FFT fft1 = new FFT(256,frequency);

        float [] localInput = new float[256];
        // Zero Pad signal
        for (int i = 0; i < FFT_SIZE; i++) {

            if (i < signal.length) {
                localInput[i] = signal[i];
            } else {
                localInput[i] = 0;
            }
        }

        fft1.forward(localInput);

       // float[] fft_cpx = fft1.getSpectrum();
        float[] imag = fft1.getImaginaryPart();
        float[] real = fft1.getRealPart();

        float[] mag = new float[FFT_SIZE];

        int highestFrequency = 0;
        float highestVal =0;
        for (int i = 0; i < FFT_SIZE; i++) {

            float val = (float)Math.sqrt((real[i] * real[i]) + (imag[i] * imag[i]));
mag[i] = val;

        }

        // ignore dc
        mag[0] =0;

        // find the highest value


        return mag;
    }

    Calendar c = Calendar.getInstance();

    //float [] rawSignal = new float[256];
    int currentWindow = 0;
    int runningSum = 0;
    float[] windowValues = new float[NUM_TAPS+FFT_SIZE];
    long windowStart = 0;
    long windowEnd = 0;

    @Override
    public Mat onCameraFrame(CameraBridgeViewBase.CvCameraViewFrame inputFrame) {
        Mat currentFrame = inputFrame.rgba();
//        Core.mean(currentFrame); -> returns x,y,z,a
        Mat gryFrame = new Mat();


        // check image size
        int cols = currentFrame.cols();
        int rows = currentFrame.rows();

        int centerCol = cols/2;
        int centerRow = rows/2;

        int avgIntensity = 0;
        int pixelCount = 0;
        for(int row=centerRow-20; row<centerRow+20; row++)
        {
            for(int col=centerCol-20; col<centerCol+20; col++)
            {
                double [] val = currentFrame.get(row,col);
                avgIntensity+=((val[0]+val[1]+val[2])/3);
                pixelCount++;
            }
        }// get rid the oldest sample in history:
        if (rawDataSeries.size() > HISTORY_SIZE) {
            rawDataSeries.removeFirst();
        }

        while(smoothedDataSeries.size()>HISTORY_SIZE) {
            smoothedDataSeries.removeFirst();
        }
        rawDataSeries.addLast(null, avgIntensity/pixelCount);


        cameraHistoryPlot.redraw();

        windowValues[currentWindow] = avgIntensity/pixelCount;
        if(currentWindow == 0)
        {


            windowStart = System.currentTimeMillis();
        }
        currentWindow++;
        if(currentWindow>=NUM_TAPS+FFT_SIZE)
        {
            windowEnd = System.currentTimeMillis();
            long windowDurationInMillis = windowEnd - windowStart;
            float cameraFrequency = (NUM_TAPS+FFT_SIZE) / (windowDurationInMillis / 1000);

            // Send the signal through band pass filter
            windowValues = applyBandPassFilter(windowValues);

            float [] filteredValues = new float[FFT_SIZE];

            // add filterd values to chart
            for(int i = 0 ; i<NUM_TAPS; i++)
            {
                smoothedDataSeries.addLast(null,0);
            }


            for(int i=NUM_TAPS; i<NUM_TAPS+FFT_SIZE; i++)
            {
                smoothedDataSeries.addLast(null,windowValues[i]);
                filteredValues[i-NUM_TAPS] = windowValues[i];
            }

            // calculate fft
            float [] fft = calculateFFT(filteredValues, cameraFrequency);
            //rawFftData =  new SimpleXYSeries("FFT");
           // rawFftData.useImplicitXVals();

            //
            int highestFrequency = 0;
            float highestVal =0;
            for (int i = 0; i < FFT_SIZE; i++) {

                if(i > 0 && fft[i]>highestVal)
                {
                    highestVal = fft[i];
                    highestFrequency = i;
                }
            }

            // Calculate mapping back to hz
            float hz = (float)(highestFrequency * (cameraFrequency/2)) / 128;

            peak = highestFrequency;
            beats = (int)(hz*60);


            runOnUiThread(new Runnable() {
                @Override
                public void run() {

                    textviewBeats.setText(Integer.toString(beats));
//stuff that updates ui

                }
            });


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
            currentWindow = 0;
        }

        //gryFrame.get();
        //Imgproc.cvtColor(currentFrame, currentFrame, Imgproc.COLOR_RGBA2GRAY);

        //Imgproc.Canny(currentFrame,currentFrame,cannyThreshold/3,cannyThreshold );
        return currentFrame;
    }

}
