// File: Phase1EnrollmentMl.cs
// Target: .NET 6+
// NuGet: Microsoft.ML
// Place this in your WebAPI project (e.g., /Services folder)
// Put enrollments.csv in <projectRoot>/App_Data/

using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.AspNetCore.Hosting;

#region EnrollmentTypePredictor

// -------------------------------------------------------------
//  ML.NET core class: trains from CSV, loads model, predicts
// -------------------------------------------------------------
public sealed class EnrollmentTypePredictor
{
    private readonly MLContext _ml;
    private ITransformer? _model;
    private PredictionEngine<EnrollmentRecord, EnrollmentPrediction>? _engine;
    private readonly object _lock = new object(); // thread-safety for Predict

    public EnrollmentTypePredictor(int seed = 7)
    {
        _ml = new MLContext(seed: seed);
    }

    // ---------- Schema classes ----------
    public sealed class EnrollmentRecord
    {
        [LoadColumn(0)] public DateTime PartA { get; set; }
        [LoadColumn(1)] public DateTime PartB { get; set; }
        [LoadColumn(2)] public string EnrollmentType { get; set; } = string.Empty;
    }

    public sealed class FeatureRow
    {
        public float A_OADate { get; set; }
        public float B_OADate { get; set; }
        public float DaysBetween { get; set; }
        public float YearsBetween { get; set; }
    }

    public sealed class EnrollmentPrediction
    {
        [ColumnName("PredictedLabel")] public string PredictedEnrollmentType { get; set; } = string.Empty;
        public float[] Score { get; set; } = Array.Empty<float>();
    }

    // ---------- Core training / loading ----------
    public void Train(string csvPath, string? saveModelPath = null)
    {
        if (!File.Exists(csvPath))
            throw new FileNotFoundException("CSV file not found.", csvPath);

        var data = _ml.Data.LoadFromTextFile<EnrollmentRecord>(
            path: csvPath, hasHeader: true, separatorChar: ',');

        var split = _ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 13);

        var pipeline =
            _ml.Transforms.CustomMapping<EnrollmentRecord, FeatureRow>(MapDatesToFeatures, contractName: "DateFeatureizer")
            .Append(_ml.Transforms.Concatenate("Features",
                nameof(FeatureRow.A_OADate),
                nameof(FeatureRow.B_OADate),
                nameof(FeatureRow.DaysBetween),
                nameof(FeatureRow.YearsBetween)))
            .Append(_ml.Transforms.Conversion.MapValueToKey(nameof(EnrollmentRecord.EnrollmentType), "Label"))
            .Append(_ml.MulticlassClassification.Trainers.SdcaMaximumEntropy("Label", "Features"))
            .Append(_ml.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedLabel"));

        _model = pipeline.Fit(split.TrainSet);

        var metrics = _ml.MulticlassClassification.Evaluate(
            _model.Transform(split.TestSet), "Label", "Score");

        Console.WriteLine($"[ML] MicroAccuracy: {metrics.MicroAccuracy:0.000}");
        Console.WriteLine($"[ML] MacroAccuracy: {metrics.MacroAccuracy:0.000}");

        _engine = _ml.Model.CreatePredictionEngine<EnrollmentRecord, EnrollmentPrediction>(_model);

        if (!string.IsNullOrWhiteSpace(saveModelPath))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(saveModelPath)!);
            _ml.Model.Save(_model, split.TrainSet.Schema, saveModelPath);
        }
    }

    public void Load(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("Model file not found.", modelPath);

        _model = _ml.Model.Load(modelPath, out _);
        _engine = _ml.Model.CreatePredictionEngine<EnrollmentRecord, EnrollmentPrediction>(_model);
    }

    public (string predictedEnrollmentType, float[] scores) Predict(string partA, string partB)
    {
        if (_engine == null)
            throw new InvalidOperationException("Model not loaded/trained.");

        if (!DateTime.TryParse(partA, out var a))
            throw new ArgumentException("PartA not a valid date.", nameof(partA));
        if (!DateTime.TryParse(partB, out var b))
            throw new ArgumentException("PartB not a valid date.", nameof(partB));

        var input = new EnrollmentRecord { PartA = a, PartB = b };
        lock (_lock)
        {
            var pred = _engine.Predict(input);
            return (pred.PredictedEnrollmentType, pred.Score ?? Array.Empty<float>());
        }
    }

    // ---------- Utility helpers ----------
    private static void MapDatesToFeatures(EnrollmentRecord input, FeatureRow output)
    {
        output.A_OADate = (float)input.PartA.ToOADate();
        output.B_OADate = (float)input.PartB.ToOADate();
        var days = (float)(input.PartB - input.PartA).TotalDays;
        output.DaysBetween = days;
        output.YearsBetween = days / 365.25f;
    }

    public static string ResolveAppDataPath(string contentRootPath, string fileName, bool ensureDir = true)
    {
        var dir = Path.Combine(contentRootPath ?? AppContext.BaseDirectory, "App_Data");
        if (ensureDir && !Directory.Exists(dir)) Directory.CreateDirectory(dir);
        return Path.Combine(dir, fileName);
    }

    public void TrainFromAppData(string contentRootPath, string csvFile = "enrollments.csv", string? modelFile = "enrollmentModel.zip")
    {
        var csvPath = ResolveAppDataPath(contentRootPath, csvFile);
        var modelPath = !string.IsNullOrWhiteSpace(modelFile)
            ? ResolveAppDataPath(contentRootPath, modelFile)
            : null;
        Train(csvPath, modelPath);
    }
}

#endregion

#region Phase1EnrollmentMl wrapper
// -------------------------------------------------------------
//  Phase1 wrapper: create instance in your constructor
// -------------------------------------------------------------
public sealed class Phase1EnrollmentMl
{
    private readonly EnrollmentTypePredictor _predictor;

    public Phase1EnrollmentMl(string contentRootPath)
    {
        _predictor = new EnrollmentTypePredictor();
        var modelPath = EnrollmentTypePredictor.ResolveAppDataPath(contentRootPath, "enrollmentModel.zip");
        var csvPath   = EnrollmentTypePredictor.ResolveAppDataPath(contentRootPath, "enrollments.csv");

        if (File.Exists(modelPath))
        {
            _predictor.Load(modelPath);
        }
        else
        {
            _predictor.TrainFromAppData(contentRootPath, "enrollments.csv", "enrollmentModel.zip");
        }
    }

    public (string predictedEnrollmentType, float[] scores) Predict(string partA, string partB)
        => _predictor.Predict(partA, partB);
}
#endregion

/*
-------------------------------------------------------------
HOW TO USE IN A CONTROLLER
-------------------------------------------------------------
[ApiController]
[Route("api/enrollment-ml")]
public class EnrollmentMlController : ControllerBase
{
    private readonly Phase1EnrollmentMl _ml;

    public EnrollmentMlController(IWebHostEnvironment env)
    {
        // Initialize ONCE for Phase1
        _ml = new Phase1EnrollmentMl(env.ContentRootPath);
    }

    [HttpGet("predict")]
    public IActionResult Predict([FromQuery] string partA, [FromQuery] string partB)
    {
        var (etype, scores) = _ml.Predict(partA, partB);
        return Ok(new { predictedEnrollmentType = etype, scores });
    }
}
-------------------------------------------------------------
*/
