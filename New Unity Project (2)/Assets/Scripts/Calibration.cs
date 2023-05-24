using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Oculus;
using UnityEngine.XR;

public class Calibration : MonoBehaviour
{
    public GameObject CalibrationObject;
    public GameObject countdownText;

    private GameObject currentObject;
    private GameObject grid;
    private GameObject edges;

    private List<Transform> gridTransforms = new List<Transform>();
    private Transform end = null;

    OVREyeGaze eyeGaze;

    // Start is called before the first frame update
    void Start()
    {
        StartCalibration();
        grid.SetActive(false);
        edges.SetActive(true); 
        StartCoroutine(calibration());
    }


    // Update is called once per frame
    void Update()
    {
        // Check if eye tracking is supported and enabled
        if (OVRManager.isHmdPresent && OVRManager.instance.isEyeTrackingEnabled)
        {
            // Get eye positions and rotations using other available components
            Vector3 eyePositionLeft = OVRManager.display.GetEyeRenderViewport(OVRPlugin.Eye.Left).position;
            Quaternion eyeRotationLeft = OVRManager.display.GetEyeRenderViewport(OVRPlugin.Eye.Left).rotation;

            Vector3 eyePositionRight = OVRManager.display.GetEyeRenderViewport(OVRPlugin.Eye.Right).position;
            Quaternion eyeRotationRight = OVRManager.display.GetEyeRenderViewport(OVRPlugin.Eye.Right).rotation;

            // Use eye tracking data as desired
            // For example, you can use eye positions and rotations for gaze-based interactions or other features

            // Print the eye positions in the console
            Debug.Log("Left Eye Position: " + eyePositionLeft);
            Debug.Log("Right Eye Position: " + eyePositionRight);
        }
        else
        {
            Debug.Log("Eye tracking is not supported or enabled.");
        }
    }

    public void StartCalibration()
    {
        Debug.Log("Calibration");

        currentObject = CalibrationObject;
        grid = currentObject.transform.Find("Positions").gameObject;
        edges = currentObject.transform.Find("Edges").gameObject;
        edges.SetActive(true);
        currentObject.SetActive(true);
        gridTransforms.Clear();
        foreach (Transform child in grid.transform)
        {
            gridTransforms.Add(child);
        }
    }

    IEnumerator calibration()
    {
        
        List<int> indices = new List<int>();
        for (int i = 0; i < 13; i++)
        {
            indices.Add(i);
        }
        int nextIndex = Random.Range(0, indices.Count-1);
        indices.Remove(nextIndex);
        end = gridTransforms[nextIndex];
        countdownText.SetActive(true);
        for (int i = 3; i > 0; i--)
        {
            countdownText.GetComponent<TextMesh>().text = i.ToString();
            yield return new WaitForSeconds(1);
        }
        countdownText.SetActive(false);
        GetComponent<Renderer>().enabled = true;
        while (indices.Count > 0)
        {
            transform.position = end.position;
            yield return new WaitForSeconds(2);
            nextIndex = indices[Random.Range(0, indices.Count-1)];
            indices.Remove(nextIndex);
            end = gridTransforms[nextIndex];
        }
        transform.position = end.position;
        yield return new WaitForSeconds(2);

        GetComponent<Renderer>().enabled = false;
        countdownText.SetActive(true);
        countdownText.GetComponent<TextMesh>().text = "Done";
        countdownText.SetActive(true);
        yield return new WaitForSeconds(1);
        countdownText.SetActive(false);
    }
}
