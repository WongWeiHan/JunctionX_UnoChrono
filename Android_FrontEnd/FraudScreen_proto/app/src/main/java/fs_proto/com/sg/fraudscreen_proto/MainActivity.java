package fs_proto.com.sg.fraudscreen_proto;

import android.Manifest;
import android.app.AlertDialog;
import android.app.Dialog;
import android.app.DialogFragment;
import android.content.DialogInterface;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Color;
import android.net.Uri;
import android.provider.Settings;
import android.speech.RecognitionListener;
import android.speech.RecognizerIntent;
import android.speech.SpeechRecognizer;
import android.support.v4.app.ActivityCompat;
import android.support.v4.content.ContextCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;
import android.widget.EditText;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import java.util.ArrayList;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {

    boolean permission_given;
    boolean isListening;
    ArrayList<String> spamWords = new ArrayList<String>();
    private String conversation;
    public static String ARG_TITLE = "WARNING";
    public static String ARG_MESSAGE = "THIS IS A FRAUD CALL!";

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        spamWords.add("credit");
        spamWords.add("password");
        spamWords.add("password");
        spamWords.add("card");
        spamWords.add("bank number");
        spamWords.add("card number");
        spamWords.add("id");
        spamWords.add("pin");
        spamWords.add("social id");
        spamWords.add("csv");



        final EditText editText = findViewById(R.id.editText);
        final TextView textView = findViewById(R.id.textView);


        if(ContextCompat.checkSelfPermission(this, Manifest.permission.READ_CONTACTS)== PackageManager.PERMISSION_GRANTED && ContextCompat.checkSelfPermission(this, Manifest.permission.GET_ACCOUNTS)== PackageManager.PERMISSION_GRANTED  )
            permission_given = true;
        else
        {
            if(ActivityCompat.shouldShowRequestPermissionRationale(this, Manifest.permission.RECORD_AUDIO))
            {
                Toast.makeText(this,"Please allow audio record",Toast.LENGTH_LONG).show();
            }
            else
            {
                ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.RECORD_AUDIO}, 1);
            }
        }

        final SpeechRecognizer mSpeechRecognizer = SpeechRecognizer.createSpeechRecognizer(this);

        final Intent mSpeechRecognizerIntent = new Intent(RecognizerIntent.ACTION_RECOGNIZE_SPEECH);
        mSpeechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE_MODEL,
                RecognizerIntent.LANGUAGE_MODEL_FREE_FORM);
        mSpeechRecognizerIntent.putExtra(RecognizerIntent.EXTRA_LANGUAGE,
                Locale.getDefault());

        mSpeechRecognizer.setRecognitionListener(new RecognitionListener() {
            @Override
            public void onReadyForSpeech(Bundle bundle) {

            }

            @Override
            public void onBeginningOfSpeech() {

            }

            @Override
            public void onRmsChanged(float v) {

            }

            @Override
            public void onBufferReceived(byte[] bytes) {

            }

            @Override
            public void onEndOfSpeech() {

            }

            @Override
            public void onError(int i) {
                Toast toast=Toast.makeText(getApplicationContext(),"ERROR",Toast.LENGTH_SHORT);
            }

            @Override
            public void onResults(Bundle bundle) {
                //getting all the matches
                ArrayList<String> matches = bundle
                        .getStringArrayList(SpeechRecognizer.RESULTS_RECOGNITION);

                if (matches != null) {
                    for (String s : matches) {
                        conversation = conversation + " " + s;
                        for (String n : spamWords) {
                            if (s.contains(n)) {
                                textView.setText(conversation);
                                editText.setText("Scam! Ending call");
                                editText.setBackgroundColor(Color.RED);
                                mSpeechRecognizer.cancel();
                                Intent homeIntent = new Intent(Intent.ACTION_MAIN);
                                homeIntent.addCategory( Intent.CATEGORY_HOME );
                                homeIntent.setFlags(Intent.FLAG_ACTIVITY_CLEAR_TOP);
                                startActivity(homeIntent);
                                return;
                            } else {
                                editText.setText("No Scam");
                                editText.setBackgroundColor(Color.GREEN);
                            }
                        }
                    }
                    mSpeechRecognizer.cancel();
                }
            }

            @Override
            public void onPartialResults(Bundle bundle) {

            }

            @Override
            public void onEvent(int i, Bundle bundle) {

            }


        });



        ImageButton imageButton = findViewById(R.id.imageButton);
        imageButton.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                mSpeechRecognizer.startListening(mSpeechRecognizerIntent);
                isListening = true;
            }

        });

    }




    @Override
    public void onRequestPermissionsResult(int requestCode,
                                           String permissions[], int[] grantResults) {
        // If request is cancelled, the result arrays are empty.
        if (grantResults.length > 0
                && grantResults[0] == PackageManager.PERMISSION_GRANTED) {

            permission_given = true;

        } else {

            Toast.makeText(MainActivity.this, "Permission denied to read your audio", Toast.LENGTH_SHORT).show();


        }
        return;

    }
}



