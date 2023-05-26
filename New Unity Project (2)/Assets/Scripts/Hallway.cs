using System.Collections;
using System.Collections.Generic;
using System.IO;
using UnityEngine;

public class Hallway : MonoBehaviour
{
    public GameObject TrackingPositions;
    public GameObject countdownText;

    private List<Transform> gridTransforms = new List<Transform>();
    private Transform start = null;    
    private Transform end = null;
    private float pathTime =5f;

    private float minDistance = 1.7f;
    private int currentIndex = 0;

    // Start is called before the first frame update
    void Start()
    {
        GetComponent<Renderer>().enabled = false;
        gridTransforms.Clear();
        foreach (Transform child in TrackingPositions.transform)
        {
            gridTransforms.Add(child);
        }
        TrackingPositions.SetActive(true);

        countdownText.GetComponent<TextMesh>().text = "hallway";
        countdownText.SetActive(true);
        TrackingPositions.SetActive(true);
        StartCoroutine(Evaluation());
    }

    // void OnEnable()
    // {
        
    // }

    // void OnDisable()
    // {
    //     countdownText.SetActive(false);
    //     if (recording)
    //     {
    //         filename = "hallway_" + System.DateTime.Now.ToString("yyyyMMddHHmmss") + ".csv";
    //         GetComponent<DataLogger>().SaveFile(filename);
    //         recording = false;
    //     }
    //     GetComponent<Renderer>().enabled = false;
    //     isEvaluating = false;
    //     movement = "start";
    //     frameNumber = 0;
    //     TrackingPositions.SetActive(false);
    // }

    // Update is called once per frame
    // void Update()
    // {
    //     if (recording)
    //     {
    //         GetComponent<DataLogger>().AddFrame(frameNumber, movement);
    //         frameNumber++;
    //     }
    //     if (Input.GetKeyDown("return") && !isEvaluating)
    //     {
    //         GetComponent<AudioSource>().Play(0);
    //         StartCoroutine(Evaluation());
    //     }
    //     /*
    //     if (Input.GetKeyDown("p"))
    //     {
    //         TrackingPositions.SetActive(!TrackingPositions.activeSelf);
    //     }
    //     */
    // }

    IEnumerator Evaluation()
    {
        start = gridTransforms[currentIndex];
        end = gridTransforms[currentIndex + 1];
        TrackingPositions.SetActive(false);
        transform.position = start.position;
        GetComponent<Renderer>().enabled = true;
        countdownText.SetActive(true);
        for (int i = 3; i > 0; i--)
        {
            countdownText.GetComponent<TextMesh>().text = i.ToString();
            yield return new WaitForSeconds(1);
        }
        countdownText.SetActive(false);
        while (currentIndex < gridTransforms.Count-1)
        {
            start = gridTransforms[currentIndex];
            end = gridTransforms[currentIndex + 1];
            float timeElapsed = 0.0f;
            while (timeElapsed < pathTime || transform.position != end.position)
            {
                transform.position = Vector3.Lerp(start.position, end.position, Mathf.Min(1, timeElapsed / pathTime));
                timeElapsed += Time.deltaTime;
                yield return null;
            }
            while (Vector3.Distance(transform.position, Camera.main.transform.position) > minDistance)
            {
                yield return null;
            }
            currentIndex++;
        }
        GetComponent<Renderer>().enabled = false;
        TrackingPositions.SetActive(true);
        countdownText.GetComponent<TextMesh>().text = "Done";
        countdownText.SetActive(true);
        yield return new WaitForSeconds(1);
        countdownText.SetActive(false);
    }
}
