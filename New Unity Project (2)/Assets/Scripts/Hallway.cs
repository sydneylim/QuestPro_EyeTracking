// using System.Collections;
// using System.Collections.Generic;
// using System.IO;
// using UnityEngine;
// using Oculus;
// using UnityEngine.XR;
// using UnityEngine.SceneManagement;


// public class Hallway : MonoBehaviour
// {
//     public GameObject TrackingPositions;
//     public GameObject taskManager;
//     public GameObject countdownText;

//     private List<Transform> gridTransforms = new List<Transform>();
//     private Transform start = null;
//     private Transform end = null;
//     private bool recording = false;
//     private string filename;
//     private string movement = "start";
//     private int frameNumber = 0;
//     private float pathTime =5f;
//     private bool isEvaluating = false;

//     private float minDistance = 1.7f;
//     private int currentIndex = 0;

//     // Start is called before the first frame update
//     void Start()
//     {
//         GetComponent<Renderer>().enabled = false;
//         gridTransforms.Clear();
//         foreach (Transform child in TrackingPositions.transform)
//         {
//             gridTransforms.Add(child);
//         }
//         TrackingPositions.SetActive(true);
//     }

//     void OnEnable()
//     {
//         countdownText.GetComponent<TextMesh>().text = "Hallway";
//         countdownText.SetActive(true);
//         TrackingPositions.SetActive(true);
//     }

//     void OnDisable()
//     {
//         countdownText.SetActive(false);
//         GetComponent<Renderer>().enabled = false;
//         isEvaluating = false;
//         movement = "start";
//         frameNumber = 0;
//         TrackingPositions.SetActive(false);
//         AsyncOperation asyncLoad = SceneManager.LoadSceneAsync("Home");

//     }

//     // Update is called once per frame
//     void Update()
//     {
//         if (recording)
//         {
//             GetComponent<CameraController>().AddFrame(frameNumber, movement);
//             frameNumber++;
//         }
//         if (Input.GetKeyDown("return") && !isEvaluating)
//         {
//             StartCoroutine(Evaluation());
//         }
//         /*
//         if (Input.GetKeyDown("p"))
//         {
//             TrackingPositions.SetActive(!TrackingPositions.activeSelf);
//         }
//         */
//     }

//     IEnumerator Evaluation()
//     {
//         isEvaluating = true;
//         filename = "hallway_" + System.DateTime.Now.ToString("yyyyMMddHHmmss") + ".csv";
//         frameNumber = 0;
//         GetComponent<DataLogger>().AddHeader();
//         recording = true;
//         currentIndex = 0;
//         start = gridTransforms[currentIndex];
//         end = gridTransforms[currentIndex + 1];
//         TrackingPositions.SetActive(false);
//         transform.position = start.position;
//         GetComponent<Renderer>().enabled = true;
//         countdownText.SetActive(true);
//         for (int i = 3; i > 0; i--)
//         {
//             countdownText.GetComponent<TextMesh>().text = i.ToString();
//             yield return new WaitForSeconds(1);
//         }
//         countdownText.SetActive(false);
//         while (currentIndex < gridTransforms.Count-1)
//         {
//             Debug.Log(currentIndex);
//             start = gridTransforms[currentIndex];
//             end = gridTransforms[currentIndex + 1];
//             float timeElapsed = 0.0f;
//             while (timeElapsed < pathTime || transform.position != end.position)
//             {
//                 movement = "moving";
//                 transform.position = Vector3.Lerp(start.position, end.position, Mathf.Min(1, timeElapsed / pathTime));
//                 timeElapsed += Time.deltaTime;
//                 yield return null;
//             }
//             movement = "static";
//             while (Vector3.Distance(transform.position, Camera.main.transform.position) > minDistance)
//             {
//                 yield return null;
//             }
//             currentIndex++;
//         }
//         GetComponent<Renderer>().enabled = false;
//         recording = false;
//         GetComponent<DataLogger>().SaveFile(filename);
//         frameNumber = 0;
//         TrackingPositions.SetActive(true);
//         isEvaluating = false;
//         countdownText.GetComponent<TextMesh>().text = "Done";
//         countdownText.SetActive(true);
//         yield return new WaitForSeconds(1);
//         countdownText.SetActive(false);
//         taskManager.GetComponent<TaskManager>().StartNextTask();
//     }
// }
