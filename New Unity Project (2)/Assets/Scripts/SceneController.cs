using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;

public class SceneController : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyDown(KeyCode.Alpha1))
        {
            StartCoroutine(LoadCalibrationScene());
        }
        else if (Input.GetKeyDown(KeyCode.Alpha2))
        {
            StartCoroutine(LoadScreenStabilizedScene());
        }
        else if (Input.GetKeyDown(KeyCode.Alpha3))
        {
            StartCoroutine(LoadWorldStabilizedScene());
        }
        else {
            Debug.Log("invalid scene number");
        }
    }

    IEnumerator LoadCalibrationScene()
    {
        AsyncOperation asyncLoad = SceneManager.LoadSceneAsync("Calibration_Arrow");

        // Wait until the asynchronous scene fully loads
        while (!asyncLoad.isDone)
        {
            yield return null;
        }
    }

    IEnumerator LoadScreenStabilizedScene()
    {
        AsyncOperation asyncLoad = SceneManager.LoadSceneAsync("ScreenStabilized_Arrow");

        // Wait until the asynchronous scene fully loads
        while (!asyncLoad.isDone)
        {
            yield return null;
        }
    }

    IEnumerator LoadWorldStabilizedScene()
    {
        AsyncOperation asyncLoad = SceneManager.LoadSceneAsync("WorldStabilized_Arrow");

        // Wait until the asynchronous scene fully loads
        while (!asyncLoad.isDone)
        {
            yield return null;
        }
    }

}
