// File: Phase1EnrollmentMl.cs
// Target: .NET 6+
// NuGet: Microsoft.ML
// CSV expected at: <projectRoot>/App_Data/enrollments.csv
//
// CSV header (required):
// PartA,PartB,EnrollmentType
// 2019-10-15,2019-11-01,AEP
// 2020-01-01,2020-01-01,ICEP
// ...

using System;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Data;

#region EnrollmentTypePredictor (trainer/loader/predictor)

public sealed class EnrollmentTypePredictor
{
    private readonly MLContext _ml;
    private ITransformer? _model;
    private PredictionEngine<EnrollmentRecord, EnrollmentPrediction>? _engine;
    private readonly object _predictLock = new object(); // PredictionEngine is not thread-safe

    public EnrollmentTypePredictor(int seed = 7)
    {
        _ml = new MLContext(seed: seed);
    }

    // ---------- Schema POCOs ----------
    public sealed class EnrollmentRecord
    {
        // CSV column 0
        [LoadColumn(0)]
        public DateTime PartA { get; set; }

        // CSV column 1
        [LoadColumn(1)]
        public DateTime PartB { get; set; }

        // CSV column 2: { ICEP, IEP, OEPI, AEP }
        [LoadColumn(2)]
        public string EnrollmentType { get; set; } = string.Empty;
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
        [ColumnName("PredictedLabel")]
        public string PredictedEnrollmentType { get; set; } = string.Empty;

        public float[] Score { get; set; } = Array.Empty<float>();
    }

    // ---------- Public API ----------
    /// <summary>
    /// Train from CSV and optionally save a model.
    /// CSV must have header: PartA,PartB,EnrollmentType
    /// </summary>
    public void Train(string csvPath, string? saveModelPath = null)
    {
        if (!File.Exists(csvPath))
            throw new FileNotFoundException("CSV file not found.", csvPath);

        // Load data
        IDataView data = _ml.Data.LoadFromTextFile<EnrollmentRecord>(
            path: csvPath,
            hasHeader: true,
            separatorChar: ',');

        // Train/test split
        var split = _ml.Data.TrainTestSplit(data, testFraction: 0.2, seed: 13);

        // Pipeline: Date -> numeric features -> Features -> Label key -> Trainer -> back to string label
        var pipeline =
            _ml.Transforms.CustomMapping<EnrollmentRecord, FeatureRow>(
                    MapDatesToFeatures, contractName: "DateFeatureizer")
            .Append(_ml.Transforms.Concatenate(
                    "Features",
                    nameof(FeatureRow.A_OADate),
                    nameof(FeatureRow.B_OADate),
                    nameof(FeatureRow.DaysBetween),
                    nameof(FeatureRow.YearsBetween)))
            // IMPORTANT: outputColumnName first, then inputColumnName
            .Append(_ml.Transforms.Conversion.MapValueToKey(
                    outputColumnName: "Label",
                    inputColumnName: nameof(EnrollmentRecord.EnrollmentType)))
            .Append(_ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(
                    labelColumnName: "Label",
                    featureColumnName: "Features"))
            // Map predicted key back to string
            .Append(_ml.Transforms.Conversion.MapKeyToValue(
                    outputColumnName: "PredictedLabel",
                    inputColumnName: "PredictedLabel"));

        // Train
        _model = pipeline.Fit(split.TrainSet);

        // Quick metrics (remove Console.WriteLine if not desired)
        var metrics = _ml.MulticlassClassification.Evaluate(
            _model.Transform(split.TestSet),
            labelColumnName: "Label",
            scoreColumnName: "Score");

        Console.WriteLine($"[ML] MicroAccuracy: {metrics.MicroAccuracy:0.000}");
        Console.WriteLine($"[ML] MacroAccuracy: {metrics.MacroAccuracy:0.000}");
        Console.WriteLine($"[ML] LogLoss:       {metrics.LogLoss:0.000}");

        // Build prediction engine
        _engine = _ml.Model.CreatePredictionEngine<EnrollmentRecord, EnrollmentPrediction>(_model);

        // Save model if asked
        if (!string.IsNullOrWhiteSpace(saveModelPath))
        {
            Directory.CreateDirectory(Path.GetDirectoryName(saveModelPath)!);
            _ml.Model.Save(_model, split.TrainSet.Schema, saveModelPath);
        }
    }

    /// <summary>
    /// Load a previously saved model (.zip).
    /// </summary>
    public void Load(string modelPath)
    {
        if (!File.Exists(modelPath))
            throw new FileNotFoundException("Model file not found.", modelPath);

        _model = _ml.Model.Load(modelPath, out _);
        _engine = _ml.Model.CreatePredictionEngine<EnrollmentRecord, EnrollmentPrediction>(_model);
    }

    /// <summary>
    /// Predict from PartA and PartB (e.g., "2021-01-01").
    /// Returns (predictedEnrollmentType, scores[]).
    /// </summary>
    public (string predictedEnrollmentType, float[] scores) Predict(string partA, string partB)
    {
        if (_engine == null)
            throw new InvalidOperationException("Model is not loaded/trained. Call Train(...) or Load(...) first.");

        if (!DateTime.TryParse(partA, out var a))
            throw new ArgumentException("PartA could not be parsed as a date.", nameof(partA));
        if (!DateTime.TryParse(partB, out var b))
            throw new ArgumentException("PartB could not be parsed as a date.", nameof(partB));

        var input = new EnrollmentRecord { PartA = a, PartB = b, EnrollmentType = string.Empty };

        // PredictionEngine is not thread-safe; guard if multiple requests can hit at once.
        lock (_predictLock)
        {
            var pred = _engine.Predict(input);
            return (pred.PredictedEnrollmentType, pred.Score ?? Array.Empty<float>());
        }
    }

    // ---------- Helpers ----------
    private static void MapDatesToFeatures(EnrollmentRecord input, FeatureRow output)
    {
        output.A_OADate = (float)input.PartA.ToOADate();
        output.B_OADate = (float)input.PartB.ToOADate();
        var days = (float)(input.PartB - input.PartA).TotalDays;
        output.DaysBetween = days;
        output.YearsBetween = days / 365.25f;
    }

    /// <summary>
    /// Resolve a file path under App_Data (creates the folder if needed).
    /// </summary>
    public static string ResolveAppDataPath(string contentRootPath, string fileName, bool ensureDir = true)
    {
        var dir = Path.Combine(contentRootPath ?? AppContext.BaseDirectory, "App_Data");
        if (ensureDir && !Directory.Exists(dir)) Directory.CreateDirectory(dir);
        return Path.Combine(dir, fileName);
    }

    /// <summary>
    /// Train using App_Data paths. Saves model to App_Data if modelFile provided.
    /// </summary>
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

#region Phase1 wrapper (instantiate in your constructor; no DI singleton)

public sealed class Phase1EnrollmentMl
{
    private readonly EnrollmentTypePredictor _predictor;

    /// <summary>
    /// Create once in your constructor. If a model exists in App_Data, it loads it;
    /// otherwise it trains from App_Data/enrollments.csv and saves App_Data/enrollmentModel.zip.
    /// </summary>
    public Phase1EnrollmentMl(string contentRootPath)
    {
        _predictor = new EnrollmentTypePredictor();

        var modelPath = EnrollmentTypePredictor.ResolveAppDataPath(contentRootPath, "enrollmentModel.zip");
        if (File.Exists(modelPath))
        {
            _predictor.Load(modelPath);
        }
        else
        {
            _predictor.TrainFromAppData(contentRootPath, "enrollments.csv", "enrollmentModel.zip");
        }
    }

    /// <summary>
    /// Call this wherever you have the Phase1EnrollmentMl instance.
    /// </summary>
    public (string predictedEnrollmentType, float[] scores) Predict(string partA, string partB)
        => _predictor.Predict(partA, partB);
}

#endregion

/*
===========================
HOW TO USE (Controller)
===========================
using Microsoft.AspNetCore.Mvc;

[ApiController]
[Route("api/enrollment-ml")]
public sealed class EnrollmentMlController : ControllerBase
{
    private readonly Phase1EnrollmentMl _ml;

    public EnrollmentMlController(IWebHostEnvironment env)
    {
        // Create the Phase1 instance ONCE for this controller instance
        _ml = new Phase1EnrollmentMl(env.ContentRootPath);
    }

    [HttpGet("predict")]
    public IActionResult Predict([FromQuery] string partA, [FromQuery] string partB)
    {
        var (etype, scores) = _ml.Predict(partA, partB);
        return Ok(new { predictedEnrollmentType = etype, scores });
    }
}
*/
