using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;
using Oculus;

public class DataLogger : MonoBehaviour
{
    private List<string> log = new List<string>();

    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        
    }

    public void AddFrame(int frameNumber, string movement)
    {
        log.Add(string.Format("{0},{1},{2}",
                System.DateTime.Now.ToString("MM/dd/yyyy HH:mm:ss.ffffff"),
                frameNumber,
                movement
          
                ));
    }

    public void AddHeader()
    {
        log.Clear();
        log.Add(string.Format("{0},{1},{2}",
                "Time",
                "Frame",
                "Movement"
            
                ));
    }

    public void SaveFile(string fileName)
    {
        //string filePath = Path.Combine(Application.dataPath, fileName);
        string filePath = Path.Combine(Application.persistentDataPath, fileName);
        Debug.Log(filePath);
        File.WriteAllLines(filePath, log);
        log.Clear();
    }
}
