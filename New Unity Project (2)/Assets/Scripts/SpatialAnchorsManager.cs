using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
 

public class SpatialAnchorsManager : MonoBehaviour
{
    [Serializable]
    public class AnchorData
    {
        public ulong spaceHandle;
        public string prefabName;
        public GameObject instantiatedObject = null;
    }
 
    private const string Object1PrefabName = "Object1";
    private const string Object2PrefabName = "Object2";
    public const ulong InvalidHandle = ulong.MaxValue;
    private Dictionary<ulong, AnchorData> m_createdAnchors;

    private void Start()
    {
        m_createdAnchors = new Dictionary<ulong, AnchorData>();
    }
 
    private void OnEnable()
    {
        OVRManager.SpatialEntitySetComponentEnabled += OVRManager_SpatialEntitySetComponentEnabled;
    }
 
    private void OnDisable()
    {
        OVRManager.SpatialEntitySetComponentEnabled -= OVRManager_SpatialEntitySetComponentEnabled;
    }

    private void Update()
    {
        bool trigger1Pressed = OVRInput.GetDown(OVRInput.Button.PrimaryIndexTrigger);
        bool trigger2Pressed = OVRInput.GetDown(OVRInput.Button.SecondaryIndexTrigger);
 
        //if the user has pressed the index trigger on one of the two controllers, generate an object in that position
        if (trigger1Pressed)
            GenerateSpatialAnchorForController(true);
 
        if (trigger2Pressed)
            GenerateSpatialAnchorForController(false);
    }

    private void GenerateSpatialAnchorForController(bool isLeft)
    {
        //get the pose of the controller in local tracking coordinates
        OVRPose controllerPose = new OVRPose()
        {
            position = OVRInput.GetLocalControllerPosition(isLeft ? OVRInput.Controller.LTouch : OVRInput.Controller.RTouch),
            orientation = OVRInput.GetLocalControllerRotation(isLeft ? OVRInput.Controller.LTouch : OVRInput.Controller.RTouch)
        };
 
        //create the information about the spatial anchor (time and position), that should have the same pose of the controller.
        //You can use the below template almost every time
        OVRPlugin.SpatialEntityAnchorCreateInfo createInfo = new OVRPlugin.SpatialEntityAnchorCreateInfo()
        {
            Time = OVRPlugin.GetTimeInSeconds(),
            BaseTracking = OVRPlugin.GetTrackingOriginType(),
            PoseInSpace = controllerPose.ToPosef() //notice that we take the pose in tracking coordinates and convert it from left handed to right handed reference system
        };
 
        //ask the runtime to create the spatial anchor.
        //The creation is instanteneous, and the identification handle is returned by the runtime inside the "ref" parameter
        ulong spaceHandle = InvalidHandle;
 
        if (OVRPlugin.SpatialEntityCreateSpatialAnchor(createInfo, ref spaceHandle))
        {
            //add the created anchor to the list of anchors
            m_createdAnchors[spaceHandle] = new AnchorData()
            {
                spaceHandle = spaceHandle,
                prefabName = isLeft ? Object1PrefabName : Object2PrefabName
            };
 
            //we don't care about the request Id for this sample. We so just keep it always at 0.
            //For more complicated applications, it could be useful to identify the callback relative to a particular request
            ulong requestId = 0;
 
            //We need to send a request to the runtime to enable the "Locatable" component of this anchor, if it is not enabled yet.
            //Until this component is assigned to the anchor, the anchor can't be tracked by the system, so we can't get its position (so it's basically useless).
            //From my experience, usually this is already enabled upon the creation of the anchor, so we first check if it is already enabled, and only if not
            //we send an activation request
            if (OVRPlugin.SpatialEntityGetComponentEnabled(ref spaceHandle, OVRPlugin.SpatialEntityComponentType.Locatable, out bool componentEnabled, out bool changePending))
            {
                //Activate the component. The operation returns immediately only an error code, but actually the request is anynchronous and gets satisfied by the runtime
                //later in the future. We will get notified about the operation completion with the OVRManager.SpatialEntitySetComponentEnabled event
                if (!componentEnabled)
                {
                    if (!OVRPlugin.SpatialEntitySetComponentEnabled(ref spaceHandle, OVRPlugin.SpatialEntityComponentType.Locatable, true, 0, ref requestId))
                        Debug.LogError("Addition of Locatable component to spatial anchor failed");
                }
                //else if it was already enabled, just create the gameobject for this anchor
                else
                {
                    GenerateOrUpdateGameObjectForAnchor(spaceHandle);
                }
            }
            else
                Debug.LogError("Get status of Locatable component to spatial anchor failed");
        }
        else
            Debug.LogError("Creation of spatial anchor failed");
    }

    private void GenerateOrUpdateGameObjectForAnchor(ulong spaceHandle)
    {
        //create the gameobject associated with the anchor, if it didn't exist
        if (m_createdAnchors[spaceHandle].instantiatedObject == null)
            m_createdAnchors[spaceHandle].instantiatedObject = GameObject.Instantiate(Resources.Load<GameObject>(m_createdAnchors[spaceHandle].prefabName));
 
        //get its pose in world space: at first we get it into headset tracking space,
        //then we convert it to world coordinates
        var anchorPose = OVRPlugin.LocateSpace(ref spaceHandle, OVRPlugin.GetTrackingOriginType());
        var anchorPoseUnity = OVRExtensions.ToWorldSpacePose(anchorPose.ToOVRPose());
 
        //assign the pose to the object associated to this anchor.
        m_createdAnchors[spaceHandle].instantiatedObject.transform.position = anchorPoseUnity.position;
        m_createdAnchors[spaceHandle].instantiatedObject.transform.rotation = anchorPoseUnity.orientation;
    }
 

    private void OVRManager_SpatialEntitySetComponentEnabled(UInt64 requestId, bool result, OVRPlugin.SpatialEntityComponentType componentType, ulong spaceHandle)
    {
        //check the operation completed successfully
        if (result)
        {
            //we should have added the data about the created anchor in the dictionary.
            //If it is not so, abort the operation
            if (!m_createdAnchors.ContainsKey(spaceHandle))
            {
                Debug.LogError("Asked to activate a component on an unknown anchor, aborting");
                return;
            }
 
            //The anchor has become Locatable, so we can actually spawn an object at its position.
            //Generate an object of the type specified in the dictionary about this anchor, and with the world pose of this anchor
            GenerateOrUpdateGameObjectForAnchor(spaceHandle);
 
            Debug.Log($"Addition of {componentType.ToString()} component to spatial anchor successfully completed");
        }
        else
            Debug.LogError($"Addition of {componentType.ToString()} component to spatial anchor failed");
    }
 
}
 