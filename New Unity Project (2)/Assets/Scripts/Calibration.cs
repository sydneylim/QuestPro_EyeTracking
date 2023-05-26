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