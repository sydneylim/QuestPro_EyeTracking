using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    public GameObject leftArrow;
    public GameObject rightArrow;
    OVRCameraRig cameraRig;
    public GameObject debugText;
    OVREyeGaze[] eyeGazes;

    void Start()
    {
        cameraRig = GetComponent<OVRCameraRig>();
        eyeGazes = cameraRig.GetComponents<OVREyeGaze>();

    }

    void Update()
    {
        if (cameraRig == null) {
            debugText.GetComponent<TextMesh>().text = "Camera Not Detected";
            return;
        }

        if (eyeGazes[0] == null) {
            debugText.GetComponent<TextMesh>().text = "Left Gaze Not Detected";
            return;
        }

        if (eyeGazes[1] == null) {
            debugText.GetComponent<TextMesh>().text = "Right Gaze Not Detected";
            return;
        }

        // camera data
        string leftEyePos = cameraRig.leftEyeAnchor.position.ToString();
        string leftEyeRot = cameraRig.leftEyeAnchor.rotation.ToString();
        string rightEyePos = cameraRig.rightEyeAnchor.position.ToString();
        string rightEyeRot = cameraRig.rightEyeAnchor.rotation.ToString();
        // headset position and rotation
        string centerEyePos = cameraRig.centerEyeAnchor.position.ToString();
        string centerEyeRot = cameraRig.centerEyeAnchor.rotation.ToString();

        string cameraDebug = "LeftEyePos: " + leftEyePos + "\n" + 
                            "LeftEyeRot: " + leftEyeRot + "\n" + 
                            "RightEyePos: " + rightEyePos + "\n" + 
                            "RightEyeRot: " + rightEyeRot + "\n" + 
                            "CenterEyePos: " + centerEyePos + "\n" + 
                            "CenterEyeRot: " + centerEyeRot;

        
        if (eyeGazes[0].EyeTrackingEnabled && eyeGazes[1].EyeTrackingEnabled) {
            // left eye gaze controller data
            // arrow for visual tracking indicator
            leftArrow.transform.rotation = eyeGazes[0].transform.rotation;
            leftArrow.transform.position = eyeGazes[0].transform.position;

            string leftName = eyeGazes[0].transform.name;
            string leftWorldRot = eyeGazes[0].transform.rotation.ToString();
            string leftWorldPos = eyeGazes[0].transform.position.ToString();
            string leftEulerAngles = eyeGazes[0].transform.eulerAngles.ToString();
            string leftForward = eyeGazes[0].transform.forward.ToString();
            string leftRight = eyeGazes[0].transform.right.ToString();
            string leftUp = eyeGazes[0].transform.up.ToString();
            string leftTrackingMode = eyeGazes[0].TrackingMode.ToString();
            string leftConfidence = eyeGazes[0].Confidence.ToString();


            string leftEyeControllerDebug = "Name: " + leftName + "\n" +
                                            "WorldRot: " + leftWorldRot + "\n" + 
                                            "WorldPos: " + leftWorldPos + "\n" + 
                                            "EulerAngles: " + leftEulerAngles + "\n" + 
                                            "ForwardVec: " + leftForward + "\n" + 
                                            "RightVec: " + leftRight + "\n" + 
                                            "UpVec: " + leftUp + "\n" + 
                                            "TrackingMode: " + leftTrackingMode + "\n" + 
                                            "Confidence: " + leftConfidence;

            // right eye gaze controller data      
            // arrow for visual tracking indicator
            rightArrow.transform.rotation = eyeGazes[1].transform.rotation;
            rightArrow.transform.position = eyeGazes[1].transform.position;

            string rightName = eyeGazes[1].transform.name;
            string rightWorldRot = eyeGazes[1].transform.rotation.ToString();
            string rightWorldPos = eyeGazes[1].transform.position.ToString();
            string rightEulerAngles = eyeGazes[1].transform.eulerAngles.ToString();
            string rightForward = eyeGazes[1].transform.forward.ToString();
            string rightRight = eyeGazes[1].transform.right.ToString();
            string rightUp = eyeGazes[1].transform.up.ToString();
            string rightTrackingMode = eyeGazes[1].TrackingMode.ToString();
            string rightConfidence = eyeGazes[1].Confidence.ToString();


            string rightEyeControllerDebug = "Name: " + rightName + "\n" +
                                            "WorldRot: " + rightWorldRot + "\n" + 
                                            "WorldPos: " + rightWorldPos + "\n" + 
                                            "EulerAngles: " + rightEulerAngles + "\n" + 
                                            "ForwardVec: " + rightForward + "\n" + 
                                            "RightVec: " + rightRight + "\n" + 
                                            "UpVec: " + rightUp + "\n" + 
                                            "TrackingMode: " + rightTrackingMode + "\n" + 
                                            "Confidence: " + rightConfidence;
            

            debugText.GetComponent<TextMesh>().text = leftEyeControllerDebug + "\n" + rightEyeControllerDebug;
        }
    }
}