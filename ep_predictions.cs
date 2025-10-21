using System;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace EnrollmentAI
{
    /// <summary>
    /// Plug-and-play ML.NET predictor for EnrollmentType + RequiredParts,
    /// with a transparent rule-based EffectiveDate recommender.
    /// - Train from CSV (header required) -> saves 2 models to App_Data
    /// - Predict from runtime input
    /// - ML.NET 3.x compatible (no OutputKind, correct label mapping)
    /// </summary>
    public sealed class EnrollmentTypePredictor
    {
        // ----- Public types ----------------------------------------------------

        public sealed class Paths
        {
            public string AppDataFolder { get; init; } =
                Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "App_Data");

            public string TrainingCsvFileName { get; init; } = "enrollment_training.csv";
            public string EnrollmentTypeModelFileName { get; init; } = "enrollmenttype-model.zip";
            public string RequiredPartsModelFileName { get; init; } = "requiredparts-model.zip";

            public string TrainingCsvFullPath => Path.Combine(AppDataFolder, TrainingCsvFileName);
            public string EnrollmentTypeModelFullPath => Path.Combine(AppDataFolder, EnrollmentTypeModelFileName);
            public string RequiredPartsModelFullPath => Path.Combine(AppDataFolder, RequiredPartsModelFileName);
        }

        public sealed class PredictInput
        {
            public string PartA { get; set; }           // e.g., "2022-05-01" or ""
            public string PartB { get; set; }           // e.g., "2022-06-01" or ""
            public string DOB { get; set; }             // required
            public string EffectiveDate { get; set; }   // optional "candidate"
            public string SEPReasonCode { get; set; }   // e.g., "LOSS_OF_COVERAGE"
            public string ProductCode { get; set; }     // "MA" | "MAPD" | "PDP"
        }

        public sealed class PredictResult
        {
            public string PredictedEnrollmentType { get; set; }   // ICEP/IEP/OEP/AEP/SEP/OEPI
            public string PredictedRequiredParts { get; set; }    // A/B/Both
            public DateTime RecommendedEffectiveDate { get; set; }
            public string Notes { get; set; }
            public float EnrollmentTypeConfidence { get; set; }   // max softmax score
            public float RequiredPartsConfidence { get; set; }    // max softmax score
        }

        // ----- Construction ----------------------------------------------------

        public EnrollmentTypePredictor(Paths paths = null)
        {
            _paths = paths ?? new Paths();
            Directory.CreateDirectory(_paths.AppDataFolder);
            _ml = new MLContext(seed: 42);
        }

        // ----- Training --------------------------------------------------------

        /// <summary>
        /// Train both models from CSV and save them to App_Data.
        /// CSV header must be:
        /// PartA,PartB,DOB,EffectiveDate,SEPReasonCode,ProductCode,EnrollmentType,RequiredParts
        /// </summary>
        public void TrainFromCsvAndSave()
        {
            EnsureFileExists(_paths.TrainingCsvFullPath);

            // Load raw rows
            var raw = _ml.Data.LoadFromTextFile<ModelRow>(_paths.TrainingCsvFullPath, hasHeader: true, separatorChar: ',');

            // Convert to strongly engineered rows (same logic as prediction)
            var rawEnum = _ml.Data.CreateEnumerable<ModelRow>(raw, reuseRowObject: false);
            var engineered = rawEnum.Select(ToFeatureTrainingRow).ToList();

            // Split engineered rows (we’ll create two IDataViews from this list)
            var rnd = new Random(42);
            var shuffled = engineered.OrderBy(_ => rnd.Next()).ToList();
            int testCount = Math.Max(1, shuffled.Count / 10);
            var test = shuffled.Take(testCount).ToList();
            var train = shuffled.Skip(testCount).ToList();

            // EnrollmentType pipeline
            var etTrain = _ml.Data.LoadFromEnumerable(train.Select(et => et.ToET()));
            var etTest  = _ml.Data.LoadFromEnumerable(test.Select(et => et.ToET()));

            var etPipeline =
                _ml.Transforms.Concatenate("NumFeat",
                        nameof(ETRow.HasPartA),
                        nameof(ETRow.HasPartB),
                        nameof(ETRow.MonthsSincePartA),
                        nameof(ETRow.MonthsSincePartB),
                        nameof(ETRow.AgeAtEffective),
                        nameof(ETRow.IsMAPD),
                        nameof(ETRow.IsMA),
                        nameof(ETRow.IsPDP))
                   .Append(_ml.Transforms.Categorical.OneHotEncoding(nameof(ETRow.SEPReasonCode))) // 3.x: no OutputKind arg
                   .Append(_ml.Transforms.Concatenate("Features", "NumFeat", nameof(ETRow.SEPReasonCode)))
                   .Append(_ml.Transforms.NormalizeMinMax("Features"))
                   .Append(_ml.Transforms.Conversion.MapValueToKey("LabelET", nameof(ETRow.EnrollmentType)))
                   .Append(_ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "LabelET", featureColumnName: "Features"))
                   .Append(_ml.Transforms.Conversion.MapKeyToValue("PredictedEnrollmentType", "PredictedLabel"))
                   .AppendCacheCheckpoint(_ml);

            var etModel = etPipeline.Fit(etTrain);
            var etPred = etModel.Transform(etTest);
            var etMetrics = _ml.MulticlassClassification.Evaluate(etPred, labelColumnName: "LabelET", predictedLabelColumnName: "PredictedLabel");

            // RequiredParts pipeline
            var rpTrain = _ml.Data.LoadFromEnumerable(train.Select(rp => rp.ToRP()));
            var rpTest  = _ml.Data.LoadFromEnumerable(test.Select(rp => rp.ToRP()));

            var rpPipeline =
                _ml.Transforms.Concatenate("NumFeat",
                        nameof(RPRow.HasPartA),
                        nameof(RPRow.HasPartB),
                        nameof(RPRow.MonthsSincePartA),
                        nameof(RPRow.MonthsSincePartB),
                        nameof(RPRow.AgeAtEffective),
                        nameof(RPRow.IsMAPD),
                        nameof(RPRow.IsMA),
                        nameof(RPRow.IsPDP))
                   .Append(_ml.Transforms.Categorical.OneHotEncoding(nameof(RPRow.SEPReasonCode)))
                   .Append(_ml.Transforms.Concatenate("Features", "NumFeat", nameof(RPRow.SEPReasonCode)))
                   .Append(_ml.Transforms.NormalizeMinMax("Features"))
                   .Append(_ml.Transforms.Conversion.MapValueToKey("LabelRP", nameof(RPRow.RequiredParts)))
                   .Append(_ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "LabelRP", featureColumnName: "Features"))
                   .Append(_ml.Transforms.Conversion.MapKeyToValue("PredictedRequiredParts", "PredictedLabel"))
                   .AppendCacheCheckpoint(_ml);

            var rpModel = rpPipeline.Fit(rpTrain);
            var rpPred = rpModel.Transform(rpTest);
            var rpMetrics = _ml.MulticlassClassification.Evaluate(rpPred, labelColumnName: "LabelRP", predictedLabelColumnName: "PredictedLabel");

            // Save models + quick metrics
            _ml.Model.Save(etModel, etTrain.Schema, _paths.EnrollmentTypeModelFullPath);
            _ml.Model.Save(rpModel, rpTrain.Schema, _paths.RequiredPartsModelFullPath);

            File.WriteAllText(Path.Combine(_paths.AppDataFolder, "metrics.txt"),
$@"EnrollmentType  MicroAcc={etMetrics.MicroAccuracy:F4}  MacroAcc={etMetrics.MacroAccuracy:F4}  LogLoss={etMetrics.LogLoss:F4}
RequiredParts    MicroAcc={rpMetrics.MicroAccuracy:F4}  MacroAcc={rpMetrics.MacroAccuracy:F4}  LogLoss={rpMetrics.LogLoss:F4}
");
        }

        // ----- Prediction ------------------------------------------------------

        public PredictResult Predict(PredictInput input)
        {
            LoadModels();

            // Build features exactly as during training
            var ft = ToFeatureRowFromPredict(input);

            // EnrollmentType
            var etEngine = _ml.Model.CreatePredictionEngine<ETRow, ETScore>(_etModel);
            var etRow = ft.ToET();
            var etScore = etEngine.Predict(etRow);

            // RequiredParts
            var rpEngine = _ml.Model.CreatePredictionEngine<RPRow, RPScore>(_rpModel);
            var rpRow = ft.ToRP();
            var rpScore = rpEngine.Predict(rpRow);

            var predictedType = etScore.PredictedEnrollmentType ?? "AEP";
            var predictedParts = rpScore.PredictedRequiredParts ??
                                 InferRequiredPartsFallback(ft.ProductCode, ft.HasPartA > 0.5f, ft.HasPartB > 0.5f);

            var (recEff, notes) = RecommendEffectiveDate(predictedType, ft._DebugParsed);

            return new PredictResult
            {
                PredictedEnrollmentType = predictedType,
                PredictedRequiredParts = predictedParts,
                RecommendedEffectiveDate = recEff,
                Notes = notes,
                EnrollmentTypeConfidence = MaxProb(etScore.Score),
                RequiredPartsConfidence = MaxProb(rpScore.Score)
            };
        }

        // ----- Internals -------------------------------------------------------

        private readonly Paths _paths;
        private readonly MLContext _ml;
        private ITransformer _etModel;
        private ITransformer _rpModel;

        private void LoadModels()
        {
            if (_etModel == null || _rpModel == null)
            {
                EnsureFileExists(_paths.EnrollmentTypeModelFullPath);
                EnsureFileExists(_paths.RequiredPartsModelFullPath);
                _etModel = _ml.Model.Load(_paths.EnrollmentTypeModelFullPath, out _);
                _rpModel = _ml.Model.Load(_paths.RequiredPartsModelFullPath, out _);
            }
        }

        private static void EnsureFileExists(string path)
        {
            if (!File.Exists(path)) throw new FileNotFoundException($"File not found: {path}");
        }

        private static float MaxProb(float[] arr) => (arr == null || arr.Length == 0) ? 0f : arr.Max();

        private static string InferRequiredPartsFallback(string productCode, bool hasA, bool hasB)
        {
            if (string.Equals(productCode, "PDP", StringComparison.OrdinalIgnoreCase))
                return "B"; // conservative default for PDP flow
            return (hasA && hasB) ? "Both" : "Both";
        }

        // ----- EffectiveDate recommendation (transparent post-rules) -----------

        private (DateTime recommended, string notes) RecommendEffectiveDate(string predictedType, Parsed p)
        {
            var baseDate = p?.EffectiveDate ?? FirstOfNextMonth(DateTime.Today);
            string note;

            switch ((predictedType ?? "AEP").ToUpperInvariant())
            {
                case "AEP":
                    var jan1 = new DateTime(baseDate.Month >= 10 ? baseDate.Year + 1 : baseDate.Year, 1, 1);
                    note = "AEP: Recommend Jan 1 effective following the election window.";
                    return (jan1, note);

                case "OEP":
                    var constrained = ConstrainToWindow(baseDate, new DateTime(baseDate.Year, 1, 1), new DateTime(baseDate.Year, 3, 31));
                    note = "OEP: First-of-next-month within Jan–Mar window.";
                    return (FirstOfNextMonth(constrained), note);

                case "IEP":
                    var sixtyFifth = new DateTime(p.DOB.Year + 65, p.DOB.Month, 1);
                    var iepEff = FirstOfNextMonth(Max(p.Today, sixtyFifth));
                    note = "IEP: First-of-next-month near 65th-birthday eligibility.";
                    return (iepEff, note);

                case "ICEP":
                    var afterB = p.PartB ?? p.PartA ?? baseDate;
                    var icepEff = FirstOfNextMonth(afterB);
                    note = "ICEP: First-of-next-month after Part B (or A if B missing).";
                    return (icepEff, note);

                case "SEP":
                    var sepEff = FirstOfNextMonth(baseDate);
                    note = $"SEP ({p.SEPReasonCode ?? "unspecified"}): First-of-next-month (simplified forward rule).";
                    return (sepEff, note);

                case "OEPI":
                    var oepiEff = FirstOfNextMonth(baseDate);
                    note = "OEPI: First-of-next-month (simplified).";
                    return (oepiEff, note);

                default:
                    return (FirstOfNextMonth(baseDate), "Defaulted: First-of-next-month.");
            }
        }

        private static DateTime FirstOfNextMonth(DateTime dt)
        {
            var y = dt.Year + (dt.Month == 12 ? 1 : 0);
            var m = dt.Month == 12 ? 1 : dt.Month + 1;
            return new DateTime(y, m, 1);
        }

        private static DateTime ConstrainToWindow(DateTime candidate, DateTime startInclusive, DateTime endInclusive)
        {
            if (candidate < startInclusive) return startInclusive;
            if (candidate > endInclusive) return endInclusive;
            return candidate;
        }

        private static DateTime Max(DateTime a, DateTime b) => a >= b ? a : b;

        // ----- Parsing + Feature engineering -----------------------------------

        private sealed class ModelRow
        {
            [LoadColumn(0)] public string PartA { get; set; }
            [LoadColumn(1)] public string PartB { get; set; }
            [LoadColumn(2)] public string DOB { get; set; }
            [LoadColumn(3)] public string EffectiveDate { get; set; }
            [LoadColumn(4)] public string SEPReasonCode { get; set; }
            [LoadColumn(5)] public string ProductCode { get; set; }
            [LoadColumn(6)] public string EnrollmentType { get; set; } // label 1
            [LoadColumn(7)] public string RequiredParts { get; set; }  // label 2
        }

        private readonly string[] _dateFormats = new[]
        {
            "M/d/yyyy","MM/dd/yyyy","M/d/yy","MM/dd/yy",
            "yyyy-MM-dd","yyyy/M/d","M-yyyy","MM-yyyy","M-yy","MM-yy",
            "M/yyyy","MM/yyyy","yyyyMMdd"
        };

        private bool TryParseDate(string value, out DateTime dt)
        {
            return DateTime.TryParseExact(value?.Trim(),
                       _dateFormats,
                       CultureInfo.InvariantCulture,
                       DateTimeStyles.AssumeLocal | DateTimeStyles.AllowWhiteSpaces,
                       out dt)
                   || DateTime.TryParse(value, out dt);
        }

        private DateTime? ParseDateOptional(string v)
        {
            if (string.IsNullOrWhiteSpace(v)) return null;
            if (TryParseDate(v, out var d)) return d;
            throw new ArgumentException($"Invalid date: '{v}'");
        }

        private DateTime ParseDateRequired(string v, string name)
        {
            if (TryParseDate(v, out var d)) return d;
            throw new ArgumentException($"Invalid {name}: '{v}'");
        }

        private FeatureTrainingRow ToFeatureTrainingRow(ModelRow r)
        {
            var parsed = new Parsed
            {
                DOB = ParseDateRequired(r.DOB, nameof(r.DOB)),
                PartA = ParseDateOptional(r.PartA),
                PartB = ParseDateOptional(r.PartB),
                EffectiveDate = ParseDateOptional(r.EffectiveDate),
                SEPReasonCode = (r.SEPReasonCode ?? "").Trim(),
                ProductCode = (r.ProductCode ?? "").Trim()
            };

            var baseDate = parsed.EffectiveDate ?? DateTime.Today;

            float monthsA = parsed.PartA.HasValue ? MonthsBetween(parsed.PartA.Value, baseDate) : 0f;
            float monthsB = parsed.PartB.HasValue ? MonthsBetween(parsed.PartB.Value, baseDate) : 0f;
            float ageEff = YearsBetween(parsed.DOB, baseDate);

            var isMA   = string.Equals(parsed.ProductCode, "MA",   StringComparison.OrdinalIgnoreCase) ? 1f : 0f;
            var isMAPD = string.Equals(parsed.ProductCode, "MAPD", StringComparison.OrdinalIgnoreCase) ? 1f : 0f;
            var isPDP  = string.Equals(parsed.ProductCode, "PDP",  StringComparison.OrdinalIgnoreCase) ? 1f : 0f;

            return new FeatureTrainingRow
            {
                HasPartA = parsed.PartA.HasValue ? 1f : 0f,
                HasPartB = parsed.PartB.HasValue ? 1f : 0f,
                MonthsSincePartA = monthsA,
                MonthsSincePartB = monthsB,
                AgeAtEffective = ageEff,
                IsMAPD = isMAPD,
                IsMA = isMA,
                IsPDP = isPDP,
                SEPReasonCode = parsed.SEPReasonCode,

                // NEW: include ProductCode so fallback can see it
                ProductCode = parsed.ProductCode,

                // Labels for training (may be null during prediction path)
                EnrollmentType = (r.EnrollmentType ?? "").Trim(),
                RequiredParts = (r.RequiredParts ?? "").Trim(),

                // Keep parsed around for EffectiveDate notes (prediction path)
                _DebugParsed = parsed
            };
        }

        private FeatureTrainingRow ToFeatureRowFromPredict(PredictInput input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            var parsed = new Parsed
            {
                DOB = ParseDateRequired(input.DOB, nameof(input.DOB)),
                PartA = ParseDateOptional(input.PartA),
                PartB = ParseDateOptional(input.PartB),
                EffectiveDate = ParseDateOptional(input.EffectiveDate),
                SEPReasonCode = (input.SEPReasonCode ?? "").Trim(),
                ProductCode = (input.ProductCode ?? "").Trim()
            };

            var baseDate = parsed.EffectiveDate ?? DateTime.Today;

            float monthsA = parsed.PartA.HasValue ? MonthsBetween(parsed.PartA.Value, baseDate) : 0f;
            float monthsB = parsed.PartB.HasValue ? MonthsBetween(parsed.PartB.Value, baseDate) : 0f;
            float ageEff = YearsBetween(parsed.DOB, baseDate);

            var isMA   = string.Equals(parsed.ProductCode, "MA",   StringComparison.OrdinalIgnoreCase) ? 1f : 0f;
            var isMAPD = string.Equals(parsed.ProductCode, "MAPD", StringComparison.OrdinalIgnoreCase) ? 1f : 0f;
            var isPDP  = string.Equals(parsed.ProductCode, "PDP",  StringComparison.OrdinalIgnoreCase) ? 1f : 0f;

            return new FeatureTrainingRow
            {
                HasPartA = parsed.PartA.HasValue ? 1f : 0f,
                HasPartB = parsed.PartB.HasValue ? 1f : 0f,
                MonthsSincePartA = monthsA,
                MonthsSincePartB = monthsB,
                AgeAtEffective = ageEff,
                IsMAPD = isMAPD,
                IsMA = isMA,
                IsPDP = isPDP,
                SEPReasonCode = parsed.SEPReasonCode,
                ProductCode = parsed.ProductCode,
                _DebugParsed = parsed
            };
        }

        // Shared numeric helpers
        private static float MonthsBetween(DateTime start, DateTime end)
        {
            var months = (end.Year - start.Year) * 12 + (end.Month - start.Month);
            if (end.Day < start.Day) months -= 1;
            return Math.Max(months, 0);
        }

        private static float YearsBetween(DateTime start, DateTime end)
        {
            var years = (end - start).TotalDays / 365.2425;
            return (float)Math.Max(0, years);
        }

        // Parsed value carrier (for recommendation)
        private sealed class Parsed
        {
            public DateTime Today { get; init; } = DateTime.Today;
            public DateTime DOB { get; init; }
            public DateTime? PartA { get; init; }
            public DateTime? PartB { get; init; }
            public DateTime? EffectiveDate { get; init; }
            public string SEPReasonCode { get; init; }
            public string ProductCode { get; init; }
        }

        // Single engineered row used during training (supplies either ET or RP view)
        private sealed class FeatureTrainingRow
        {
            public float HasPartA { get; set; }
            public float HasPartB { get; set; }
            public float MonthsSincePartA { get; set; }
            public float MonthsSincePartB { get; set; }
            public float AgeAtEffective { get; set; }
            public float IsMAPD { get; set; }
            public float IsMA { get; set; }
            public float IsPDP { get; set; }
            public string SEPReasonCode { get; set; }

            // Needed for fallback logic
            public string ProductCode { get; set; }

            // Labels (when training)
            public string EnrollmentType { get; set; }
            public string RequiredParts { get; set; }

            // Keep parsed around for EffectiveDate notes (prediction path)
            public Parsed _DebugParsed { get; set; }

            // Convert to ET schema row
            public ETRow ToET() => new ETRow
            {
                HasPartA = HasPartA,
                HasPartB = HasPartB,
                MonthsSincePartA = MonthsSincePartA,
                MonthsSincePartB = MonthsSincePartB,
                AgeAtEffective = AgeAtEffective,
                IsMAPD = IsMAPD,
                IsMA = IsMA,
                IsPDP = IsPDP,
                SEPReasonCode = SEPReasonCode,
                EnrollmentType = EnrollmentType,
                _Parsed = _DebugParsed,
                ProductCode = ProductCode
            };

            // Convert to RP schema row
            public RPRow ToRP() => new RPRow
            {
                HasPartA = HasPartA,
                HasPartB = HasPartB,
                MonthsSincePartA = MonthsSincePartA,
                MonthsSincePartB = MonthsSincePartB,
                AgeAtEffective = AgeAtEffective,
                IsMAPD = IsMAPD,
                IsMA = IsMA,
                IsPDP = IsPDP,
                SEPReasonCode = SEPReasonCode,
                RequiredParts = RequiredParts,
                _Parsed = _DebugParsed,
                ProductCode = ProductCode
            };
        }

        // Two concrete schema classes (so each pipeline has a clean label column)

        private sealed class ETRow
        {
            public float HasPartA { get; set; }
            public float HasPartB { get; set; }
            public float MonthsSincePartA { get; set; }
            public float MonthsSincePartB { get; set; }
            public float AgeAtEffective { get; set; }
            public float IsMAPD { get; set; }
            public float IsMA { get; set; }
            public float IsPDP { get; set; }
            public string SEPReasonCode { get; set; }

            // Label for EnrollmentType pipeline
            public string EnrollmentType { get; set; }

            // Extras for post-rule
            public Parsed _Parsed { get; set; }
            public string ProductCode { get; set; }
        }

        private sealed class RPRow
        {
            public float HasPartA { get; set; }
            public float HasPartB { get; set; }
            public float MonthsSincePartA { get; set; }
            public float MonthsSincePartB { get; set; }
            public float AgeAtEffective { get; set; }
            public float IsMAPD { get; set; }
            public float IsMA { get; set; }
            public float IsPDP { get; set; }
            public string SEPReasonCode { get; set; }

            // Label for RequiredParts pipeline
            public string RequiredParts { get; set; }

            // Extras for post-rule
            public Parsed _Parsed { get; set; }
            public string ProductCode { get; set; }
        }

        // Scores returned by PredictionEngine
        private sealed class ETScore
        {
            [ColumnName("PredictedEnrollmentType")] public string PredictedEnrollmentType { get; set; }
            public float[] Score { get; set; }
        }
        private sealed class RPScore
        {
            [ColumnName("PredictedRequiredParts")] public string PredictedRequiredParts { get; set; }
            public float[] Score { get; set; }
        }
    }
}
