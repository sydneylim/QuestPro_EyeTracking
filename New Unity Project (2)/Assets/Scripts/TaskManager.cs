using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;

public class TaskManager : MonoBehaviour
{
    public GameObject screenTrackingSphere;
    //public GameObject worldTrackingSphere;

    private static List<int> tasks;

    private List<int> order = new List<int>() { 1, 2, 3, 4, 5, 6 };

    
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        if(Input.GetKeyDown("1")) {
            StartCalibrationTask();
        }
    }

    public void DisableEverything()
    {
        screenTrackingSphere.GetComponent<Calibration>().enabled = false;
        screenTrackingSphere.SetActive(false);
        //worldTrackingSphere.SetActive(false);
    }

    public void StartCalibrationTask()
    {
        DisableEverything();
        screenTrackingSphere.SetActive(true);
        screenTrackingSphere.GetComponent<Calibration>().enabled = true;
    }
}
