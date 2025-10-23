// File: Phase1EnrollmentMl.cs
// Purpose: Create an instance in *your* constructor (no DI singleton).
// Usage example shown below.

using System.IO;
using Microsoft.AspNetCore.Hosting;

public sealed class Phase1EnrollmentMl
{
    // The underlying ML.NET predictor you already have
    private readonly EnrollmentTypePredictor _predictor;

    /// <summary>
    /// Construct once for Phase1. Loads model from App_Data if present,
    /// otherwise trains from App_Data/enrollments.csv and saves to App_Data.
    /// Pass IWebHostEnvironment.ContentRootPath from your constructor.
    /// </summary>
    public Phase1EnrollmentMl(string contentRootPath)
    {
        // Step 1: create predictor
        _predictor = new EnrollmentTypePredictor();

        // Step 2: resolve App_Data paths
        var modelPath = EnrollmentTypePredictor.ResolveAppDataPath(contentRootPath, "enrollmentModel.zip");
        var csvPath   = EnrollmentTypePredictor.ResolveAppDataPath(contentRootPath, "enrollments.csv");

        // Step 3: load if model exists; else train once and save
        if (File.Exists(modelPath))
        {
            _predictor.Load(modelPath); // fast path
        }
        else
        {
            // Train from CSV in App_Data and save a model for next runs
            _predictor.TrainFromAppData(contentRootPath, "enrollments.csv", "enrollmentModel.zip");
        }
    }

    /// <summary>
    /// Make a prediction using PartA and PartB (date strings, e.g., "2021-01-01").
    /// Returns (predictedEnrollmentType, scores[]).
    /// </summary>
    public (string predictedEnrollmentType, float[] scores) Predict(string partA, string partB)
        => _predictor.Predict(partA, partB);
}
