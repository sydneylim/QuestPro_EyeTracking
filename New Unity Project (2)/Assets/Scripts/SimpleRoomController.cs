using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SimpleRoomController : MonoBehaviour
{
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
         if (Input.GetKeyDown("up"))
        {
            transform.position += transform.up * 0.05f;
        }
        if (Input.GetKeyDown("down"))
        {
            transform.position -= transform.up * 0.05f;
        }
        if (Input.GetKeyDown("left"))
        {
            transform.position -= transform.right * 0.05f;
        }
        if (Input.GetKeyDown("right"))
        {
            transform.position += transform.right * 0.05f;
        }
        if (Input.GetKeyDown("w"))
        {
            transform.position += transform.forward * 0.05f;
        }
        if (Input.GetKeyDown("s"))
        {
            transform.position -= transform.forward * 0.05f; 
        }
        if (Input.GetKeyDown("a"))
        {
            transform.Rotate(0.0f, -1.0f, 0.0f, Space.World);
        }
        if (Input.GetKeyDown("d"))
        {
            transform.Rotate(0.0f, 1.0f, 0.0f, Space.World);
        }
        
        if (Input.GetKeyDown("n"))
        {
            transform.localScale += new Vector3(1,1,1) * 0.01f;
        }
        if (Input.GetKeyDown("m"))
        {
            transform.localScale -= new Vector3(1,1,1) * 0.01f;
        }
    }
}
