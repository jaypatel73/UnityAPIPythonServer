// ModelRunner.cs
using UnityEngine;
using System;
using System.Text;
using UnityEngine.Networking;
using System.Collections;

[Serializable]
public class GenerateRequest
{
    public string prompt;
    public int max_length;
    public float temperature;
    public bool stream;
}

[Serializable]
public class GenerateResponse
{
    public string text;
}

public class ModelRunner : MonoBehaviour
{
    private string apiUrl = "http://localhost:8000/generate";
    private string question = "What is the capital of France? Answer in one sentence.";
    private float startTime;

    void Start()
    {
        StartCoroutine(MakeAPICall());
    }

    IEnumerator MakeAPICall()
    {
        startTime = Time.realtimeSinceStartup;
        Debug.Log($"Sending question: {question}");

        GenerateRequest requestData = new GenerateRequest
        {
            prompt = question,
            max_length = 50,
            temperature = 0.7f,
            stream = false
        };

        string jsonData = JsonUtility.ToJson(requestData);
        byte[] bodyRaw = Encoding.UTF8.GetBytes(jsonData);

        UnityWebRequest request = new UnityWebRequest(apiUrl, "POST");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        yield return request.SendWebRequest();

        float elapsedTime = Time.realtimeSinceStartup - startTime;

        if (request.result == UnityWebRequest.Result.Success)
        {
            GenerateResponse response = JsonUtility.FromJson<GenerateResponse>(request.downloadHandler.text);
            Debug.Log($"Time taken: {elapsedTime:F2} seconds");
            Debug.Log($"Answer: {response.text}");
        }
        else
        {
            Debug.LogError($"Error: {request.error}");
        }

        request.Dispose();
    }
}