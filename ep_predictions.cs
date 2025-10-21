using System;
using System.Globalization;
using System.IO;
using System.Linq;
using Microsoft.ML;
using Microsoft.ML.Data;

namespace EnrollmentAI
{
    /// <summary>
    /// Plug-and-play predictor for Medicare enrollment categorization using ML.NET.
    /// - Train from CSV (header required) -> saves 2 models (EnrollmentType + RequiredParts)
    /// - Load models and predict from a single input payload
    /// - Recommends EffectiveDate using simple rules based on predicted EnrollmentType
    /// </summary>
    public sealed class EnrollmentTypePredictor
    {
        // ======= Public API ====================================================

        /// <summary>
        /// Change these if you want different locations.
        /// </summary>
        public sealed class Paths
        {
            public string AppDataFolder { get; init; } = Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "App_Data");
            public string TrainingCsvFileName { get; init; } = "enrollment_training.csv";
            public string EnrollmentTypeModelFileName { get; init; } = "enrollmenttype-model.zip";
            public string RequiredPartsModelFileName { get; init; } = "requiredparts-model.zip";

            public string TrainingCsvFullPath => Path.Combine(AppDataFolder, TrainingCsvFileName);
            public string EnrollmentTypeModelFullPath => Path.Combine(AppDataFolder, EnrollmentTypeModelFileName);
            public string RequiredPartsModelFullPath => Path.Combine(AppDataFolder, RequiredPartsModelFileName);
        }

        /// <summary>
        /// Input payload for prediction.
        /// </summary>
        public sealed class PredictInput
        {
            public string PartA { get; set; }           // e.g., "2022-05-01" or ""
            public string PartB { get; set; }           // e.g., "2022-06-01" or ""
            public string DOB { get; set; }             // required
            public string EffectiveDate { get; set; }   // candidate / requested
            public string SEPReasonCode { get; set; }   // e.g., "LOSS_OF_COVERAGE"
            public string ProductCode { get; set; }     // "MA" | "MAPD" | "PDP"
        }

        /// <summary>
        /// Final combined prediction.
        /// </summary>
        public sealed class PredictResult
        {
            public string PredictedEnrollmentType { get; set; }
            public string PredictedRequiredParts { get; set; } // "A" | "B" | "Both"
            public DateTime RecommendedEffectiveDate { get; set; }
            public string Notes { get; set; } // Any post-rule notes or reasons
            public float EnrollmentTypeConfidence { get; set; } // max score prob-ish
            public float RequiredPartsConfidence { get; set; }  // max score prob-ish
        }

        public EnrollmentTypePredictor(Paths paths = null)
        {
            _paths = paths ?? new Paths();
            Directory.CreateDirectory(_paths.AppDataFolder);

            _ml = new MLContext(seed: 42);
        }

        /// <summary>
        /// Train both models from CSV and save to disk.
        /// </summary>
        public void TrainFromCsvAndSave()
        {
            EnsureFileExists(_paths.TrainingCsvFullPath);

            var data = _ml.Data.LoadFromTextFile<ModelRow>(
                _paths.TrainingCsvFullPath,
                hasHeader: true,
                separatorChar: ',');

            // Split
            var split = _ml.Data.TrainTestSplit(data, testFraction: 0.1, seed: 42);

            // EnrollmentType pipeline
            var etPipeline = BuildFeaturizer()
                .Append(_ml.Transforms.Conversion.MapValueToKey(nameof(ModelRow.EnrollmentType), nameof(ModelRow.EnrollmentType)))
                .Append(_ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: nameof(ModelRow.EnrollmentType), featureColumnName: "Features"))
                .Append(_ml.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedEnrollmentType"))
                .AppendCacheCheckpoint(_ml);

            var etModel = etPipeline.Fit(split.TrainSet);

            // Evaluate EnrollmentType
            var etPred = etModel.Transform(split.TestSet);
            var etMetrics = _ml.MulticlassClassification.Evaluate(etPred, labelColumnName: nameof(ModelRow.EnrollmentType), predictedLabelColumnName: "PredictedLabel");

            // RequiredParts pipeline
            var rpPipeline = BuildFeaturizer()
                .Append(_ml.Transforms.Conversion.MapValueToKey(nameof(ModelRow.RequiredParts), nameof(ModelRow.RequiredParts)))
                .Append(_ml.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: nameof(ModelRow.RequiredParts), featureColumnName: "Features"))
                .Append(_ml.Transforms.Conversion.MapKeyToValue("PredictedLabel", "PredictedRequiredParts"))
                .AppendCacheCheckpoint(_ml);

            var rpModel = rpPipeline.Fit(split.TrainSet);

            // Evaluate RequiredParts
            var rpPred = rpModel.Transform(split.TestSet);
            var rpMetrics = _ml.MulticlassClassification.Evaluate(rpPred, labelColumnName: nameof(ModelRow.RequiredParts), predictedLabelColumnName: "PredictedLabel");

            // Save models
            _ml.Model.Save(etModel, split.TrainSet.Schema, _paths.EnrollmentTypeModelFullPath);
            _ml.Model.Save(rpModel, split.TrainSet.Schema, _paths.RequiredPartsModelFullPath);

            // Optional: write metrics summary beside models (helps debugging)
            File.WriteAllText(Path.Combine(_paths.AppDataFolder, "metrics.txt"),
$@"EnrollmentType  MicroAcc={etMetrics.MicroAccuracy:F4}  MacroAcc={etMetrics.MacroAccuracy:F4}  LogLoss={etMetrics.LogLoss:F4}
RequiredParts    MicroAcc={rpMetrics.MicroAccuracy:F4}  MacroAcc={rpMetrics.MacroAccuracy:F4}  LogLoss={rpMetrics.LogLoss:F4}
");
        }

        /// <summary>
        /// Predict EnrollmentType + RequiredParts and compute a recommended EffectiveDate.
        /// </summary>
        public PredictResult Predict(PredictInput input)
        {
            LoadOrThrow();

            var fe = FeatureFromInput(input, out var parsed);

            // EnrollmentType
            var etEngine = _ml.Model.CreatePredictionEngine<FeatureRow, EnrollmentTypeScore>(_etModel);
            var etScore = etEngine.Predict(fe);

            // RequiredParts
            var rpEngine = _ml.Model.CreatePredictionEngine<FeatureRow, RequiredPartsScore>(_rpModel);
            var rpScore = rpEngine.Predict(fe);

            var enrollmentType = etScore.PredictedEnrollmentType ?? "AEP";
            var requiredParts  = rpScore.PredictedRequiredParts ?? InferRequiredPartsFallback(parsed.ProductCode, parsed.HasPartA, parsed.HasPartB);

            // Post-rule recommended EffectiveDate (simple, transparent)
            var (recEff, notes) = RecommendEffectiveDate(enrollmentType, parsed);

            return new PredictResult
            {
                PredictedEnrollmentType = enrollmentType,
                PredictedRequiredParts  = requiredParts,
                RecommendedEffectiveDate = recEff,
                Notes = notes,
                EnrollmentTypeConfidence = MaxProb(etScore.Score),
                RequiredPartsConfidence  = MaxProb(rpScore.Score)
            };
        }

        // ======= Internals =====================================================

        private readonly Paths _paths;
        private readonly MLContext _ml;
        private ITransformer _etModel;
        private ITransformer _rpModel;

        private void LoadOrThrow()
        {
            if (_etModel == null || _rpModel == null)
            {
                EnsureFileExists(_paths.EnrollmentTypeModelFullPath);
                EnsureFileExists(_paths.RequiredPartsModelFullPath);

                _etModel = _ml.Model.Load(_paths.EnrollmentTypeModelFullPath, out _);
                _rpModel = _ml.Model.Load(_paths.RequiredPartsModelFullPath, out _);
            }
        }

        private static float MaxProb(float[] arr) => (arr == null || arr.Length == 0) ? 0f : arr.Max();

        private static string InferRequiredPartsFallback(string productCode, bool hasA, bool hasB)
        {
            // Conservative: MA/MAPD typically require A+B; PDP typically needs Part D eligibility (assume B sufficient in many org flows)
            if (string.Equals(productCode, "PDP", StringComparison.OrdinalIgnoreCase))
                return hasB ? "B" : "B"; // default to B
            return (hasA && hasB) ? "Both" : "Both";
        }

        private (DateTime recommended, string notes) RecommendEffectiveDate(string predictedType, ParsedInput p)
        {
            // Baseline: if user supplied a candidate EffectiveDate, start from it; else use first-of-next-month.
            var baseDate = p.EffectiveDate ?? FirstOfNextMonth(DateTime.Today);

            string note;

            switch ((predictedType ?? "AEP").ToUpperInvariant())
            {
                case "AEP":
                    // Annual Enrollment Period: Oct 15–Dec 7; most plans effect Jan 1 following year.
                    var jan1 = new DateTime(baseDate.Month >= 10 ? baseDate.Year + 1 : baseDate.Year, 1, 1);
                    note = "AEP: Recommend Jan 1 effective following the election window.";
                    return (jan1, note);

                case "OEP":
                    // Open Enrollment Period (Jan–Mar). Use first-of-next-month within window.
                    var constrained = ConstrainToWindow(baseDate, new DateTime(baseDate.Year, 1, 1), new DateTime(baseDate.Year, 3, 31));
                    note = "OEP: First-of-next-month within Jan–Mar window.";
                    return (FirstOfNextMonth(constrained), note);

                case "IEP":
                    // Initial Enrollment Period: typically first-of-month following election when first eligible around 65th birthday
                    var sixtyFifth = new DateTime(p.DOB.Year + 65, p.DOB.Month, 1);
                    var iepEff = FirstOfNextMonth(Max(p.Today, sixtyFifth));
                    note = "IEP: First-of-next-month near 65th-birthday eligibility.";
                    return (iepEff, note);

                case "ICEP":
                    // Initial Coverage Election Period (Part B activation for MA/MAPD). Often first-of-month after Part B effective.
                    var afterB = p.PartB ?? p.PartA ?? baseDate;
                    var icepEff = FirstOfNextMonth(afterB);
                    note = "ICEP: First-of-next-month after Part B (or Part A if B missing).";
                    return (icepEff, note);

                case "SEP":
                    // Special Enrollment Period: often first-of-next-month (some reasons allow retro; keeping forward-simple)
                    var sepEff = FirstOfNextMonth(baseDate);
                    note = $"SEP ({p.SEPReasonCode ?? "unspecified"}): First-of-next-month (simple forward rule).";
                    return (sepEff, note);

                case "OEPI":
                    // Institutionalized OEP: permissive; simple forward month.
                    var oepiEff = FirstOfNextMonth(baseDate);
                    note = "OEPI: First-of-next-month (simplified).";
                    return (oepiEff, note);

                default:
                    var def = FirstOfNextMonth(baseDate);
                    return (def, "Defaulted: First-of-next-month.");
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

        private static void EnsureFileExists(string path)
        {
            if (!File.Exists(path))
                throw new FileNotFoundException($"File not found: {path}");
        }

        // ======= Data contracts for ML ========================================

        private sealed class ModelRow
        {
            // Raw CSV columns
            [LoadColumn(0)] public string PartA { get; set; }
            [LoadColumn(1)] public string PartB { get; set; }
            [LoadColumn(2)] public string DOB { get; set; }
            [LoadColumn(3)] public string EffectiveDate { get; set; }
            [LoadColumn(4)] public string SEPReasonCode { get; set; }
            [LoadColumn(5)] public string ProductCode { get; set; }

            // Labels (for training)
            [LoadColumn(6)] public string EnrollmentType { get; set; } // ICEP/IEP/OEP/AEP/SEP/OEPI
            [LoadColumn(7)] public string RequiredParts { get; set; }  // A/B/Both
        }

        private sealed class FeatureRow
        {
            // Engineered numeric features
            public float HasPartA { get; set; }
            public float HasPartB { get; set; }
            public float MonthsSincePartA { get; set; }
            public float MonthsSincePartB { get; set; }
            public float AgeAtEffective { get; set; }
            public float IsMAPD { get; set; }
            public float IsMA { get; set; }
            public float IsPDP { get; set; }

            // Categorical keys
            public string SEPReasonCode { get; set; }

            // Vector
            public float[] Features { get; set; }
        }

        private sealed class EnrollmentTypeScore
        {
            [ColumnName("PredictedEnrollmentType")]
            public string PredictedEnrollmentType { get; set; }

            public float[] Score { get; set; }
        }

        private sealed class RequiredPartsScore
        {
            [ColumnName("PredictedRequiredParts")]
            public string PredictedRequiredParts { get; set; }

            public float[] Score { get; set; }
        }

        private EstimatorChain<Microsoft.ML.Transforms.NormalizingTransformer> BuildFeaturizer()
        {
            // Compose numeric features + one-hot for SEPReason
            return _ml.Transforms.Concatenate("NumFeat",
                    nameof(FeatureRow.HasPartA),
                    nameof(FeatureRow.HasPartB),
                    nameof(FeatureRow.MonthsSincePartA),
                    nameof(FeatureRow.MonthsSincePartB),
                    nameof(FeatureRow.AgeAtEffective),
                    nameof(FeatureRow.IsMAPD),
                    nameof(FeatureRow.IsMA),
                    nameof(FeatureRow.IsPDP))
                .Append(_ml.Transforms.Categorical.OneHotEncoding(nameof(FeatureRow.SEPReasonCode), outputKind: Microsoft.ML.Transforms.OutputKind.Indicator))
                .Append(_ml.Transforms.Concatenate("Features", "NumFeat", nameof(FeatureRow.SEPReasonCode)))
                .Append(_ml.Transforms.NormalizeMinMax("Features"));
        }

        // ======= Feature engineering ==========================================

        private readonly string[] _dateFormats = new[]
        {
            "M/d/yyyy","MM/dd/yyyy","M/d/yy","MM/dd/yy",
            "yyyy-MM-dd","yyyy/M/d","M-yyyy","MM-yyyy","M-yy","MM-yy",
            "M/yyyy","MM/yyyy","M-yy","MM-yy","yyyyMMdd",
        };

        private FeatureRow FeatureFromInput(PredictInput input, out ParsedInput parsed)
        {
            parsed = ParseInput(input);

            var isMA   = string.Equals(parsed.ProductCode, "MA",   StringComparison.OrdinalIgnoreCase) ? 1f : 0f;
            var isMAPD = string.Equals(parsed.ProductCode, "MAPD", StringComparison.OrdinalIgnoreCase) ? 1f : 0f;
            var isPDP  = string.Equals(parsed.ProductCode, "PDP",  StringComparison.OrdinalIgnoreCase) ? 1f : 0f;

            float monthsA = parsed.PartA.HasValue ? MonthsBetween(parsed.PartA.Value, parsed.EffectiveDate ?? parsed.Today) : 0f;
            float monthsB = parsed.PartB.HasValue ? MonthsBetween(parsed.PartB.Value, parsed.EffectiveDate ?? parsed.Today) : 0f;

            float ageEff = YearsBetween(parsed.DOB, parsed.EffectiveDate ?? parsed.Today);

            var fr = new FeatureRow
            {
                HasPartA = parsed.HasPartA ? 1f : 0f,
                HasPartB = parsed.HasPartB ? 1f : 0f,
                MonthsSincePartA = monthsA,
                MonthsSincePartB = monthsB,
                AgeAtEffective = ageEff,
                IsMA = isMA,
                IsMAPD = isMAPD,
                IsPDP = isPDP,
                SEPReasonCode = parsed.SEPReasonCode ?? "UNKNOWN"
            };

            // The pipeline will build Features.
            return fr;
        }

        private sealed class ParsedInput
        {
            public DateTime Today { get; init; } = DateTime.Today;
            public DateTime DOB { get; init; }
            public DateTime? PartA { get; init; }
            public DateTime? PartB { get; init; }
            public DateTime? EffectiveDate { get; init; }
            public string SEPReasonCode { get; init; }
            public string ProductCode { get; init; }
            public bool HasPartA => PartA.HasValue;
            public bool HasPartB => PartB.HasValue;
        }

        private ParsedInput ParseInput(PredictInput input)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            var dob = ParseDateRequired(input.DOB, nameof(input.DOB));

            return new ParsedInput
            {
                DOB = dob,
                PartA = ParseDateOptional(input.PartA),
                PartB = ParseDateOptional(input.PartB),
                EffectiveDate = ParseDateOptional(input.EffectiveDate),
                SEPReasonCode = (input.SEPReasonCode ?? "").Trim(),
                ProductCode = (input.ProductCode ?? "").Trim()
            };
        }

        private DateTime ParseDateRequired(string value, string name)
        {
            if (TryParseDate(value, out var dt)) return dt;
            throw new ArgumentException($"Invalid {name} date format: '{value}'");
        }

        private DateTime? ParseDateOptional(string value)
        {
            if (string.IsNullOrWhiteSpace(value)) return null;
            if (TryParseDate(value, out var dt)) return dt;
            // If provided but unparsable, throw to keep behavior explicit
            throw new ArgumentException($"Invalid date format: '{value}'");
        }

        private bool TryParseDate(string value, out DateTime dt)
        {
            return DateTime.TryParseExact(value?.Trim(),
                _dateFormats,
                CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeLocal | DateTimeStyles.AllowWhiteSpaces,
                out dt)
                || DateTime.TryParse(value, out dt);
        }

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
    }
}
