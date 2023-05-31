using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class EyeGazeController : MonoBehaviour
{
    public GameObject arrow;
    OVREyeGaze eyeGaze;
    public GameObject debugText;

    void Start()
    {
        eyeGaze = GetComponent<OVREyeGaze>();
    }

    void Update()
    {
          if (eyeGaze == null) {
            debugText.GetComponent<TextMesh>().text = "Gaze Not Detected";
            return;
        }

        if (eyeGaze.EyeTrackingEnabled) {
            arrow.transform.rotation = eyeGaze.transform.rotation;
            arrow.transform.position = eyeGaze.transform.position;

            debugText.GetComponent<TextMesh>().text = "Eye Tracking Enabled";
            string rotation = eyeGaze.transform.rotation.ToString();
            string position = eyeGaze.transform.position.ToString();

            debugText.GetComponent<TextMesh>().text = "rot: " + rotation + "\n" + "pos: " + position + "\n" + eyeGaze.TrackingMode + "\n" + eyeGaze.Confidence + "\n" + eyeGaze.ConfidenceThreshold + "\n" + eyeGaze.ReferenceFrame;
        }
    }
}