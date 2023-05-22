using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Reflection.Emit;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
using System.Runtime.Serialization.Formatters.Binary;
using System.Text;
using System.Threading.Tasks;

namespace MLP_TAKE2
{
    internal class MLP
    {
        private double[] currentSample;

        // Desired inputs in format:
        // First index is the index of output neuron
        // Second index is the index of sample
        private double[] desiredOutput;

        // Hidden layer and output layer matrices are in this form:
        // First index is the index of neuron
        // second index is the index of weights
        private double[][,] hiddenLayerMatrix;
        private double[,] outputLayerMatrix;

        private double[][,] hiddenLayerGradientMatrix;
        private double[,] outputLayerGradientMatrix;


        private double[][] hiddenLayerFactors;


        private double[][,] hiddenLayerDeltaMatrix;
        private double[,] outputLayerDeltaMatrix;

        private double[][] hiddenLayerBiasGradient;
        private double[] outputLayerBiasGradient;

        private double[][] hiddenLayerBiasDelta;
        private double[] outputLayerBiasDelta;


        private double[][] hiddenLayerBias;
        private double[] outputLayerBias;

        
        private int numberOfInputNeurons;
        private int numberOfOutputNeurons; //make it as a List with length of amount of hidden layers
        private int[] numberOfHiddenNeurons;
        private int numberOfHiddenLayers;


        private double[][] weighedSumsHiddenLayer;
        private double[][] hiddenLayerOutputs;

        private double[] weighedSumsOutputLayer;
        private double[] outputLayerOutputs;
        private double[] outputLayerFactor;



        private List<List<double>> data;
        private List<List<double>> desiredOutputs;

        public MLP(int numberOfInputNeurons, int numberOfHiddenLayers, List<int>numberOfHiddenNeurons,int numberOfOutputs) //(list<int> layers=[inputammout, hiddenammount1, hiddenammount2, ..., hiddenammountn, outputammount], int bias)
        {
            this.numberOfInputNeurons = numberOfInputNeurons; //layers[0]
            this.numberOfHiddenNeurons = new int[numberOfHiddenLayers];//for i=1; i<layers.length-1; i++
            for(int i = 0; i< numberOfHiddenNeurons.Count; i++)
            {
                this.numberOfHiddenNeurons[i] = numberOfHiddenNeurons[i];
            }
            this.numberOfOutputNeurons = numberOfOutputs; //layers[-1]
            this.numberOfHiddenLayers = numberOfHiddenLayers;
            desiredOutput = new double[numberOfOutputs];
            currentSample = new double[numberOfInputNeurons];

            hiddenLayerMatrix = new double[numberOfHiddenLayers][,];
            hiddenLayerGradientMatrix = new double[numberOfHiddenLayers][,];
            hiddenLayerDeltaMatrix = new double[numberOfHiddenLayers][,];
            hiddenLayerBias = new double[numberOfHiddenLayers][];
            hiddenLayerBiasGradient = new double[numberOfHiddenLayers][];
            weighedSumsHiddenLayer = new double[numberOfHiddenLayers][];
            hiddenLayerOutputs = new double[numberOfHiddenLayers][];
            hiddenLayerBiasDelta = new double[numberOfHiddenLayers][];
            hiddenLayerFactors = new double[numberOfHiddenLayers][];
            //FIRST HIDDEN LAYER HAS WEIGHTS DEPENDENT ON INPUT LAYER, THAT'S WHY THIS LINE IS HERE
            hiddenLayerMatrix[0] = new double[numberOfHiddenNeurons[0],numberOfInputNeurons];
            hiddenLayerBias[0] = new double[numberOfHiddenNeurons[0]];
            hiddenLayerBiasGradient[0] = new double[numberOfHiddenNeurons[0]];
            hiddenLayerGradientMatrix[0] = new double[numberOfHiddenNeurons[0], numberOfInputNeurons];
            hiddenLayerDeltaMatrix[0] = new double[numberOfHiddenNeurons[0], numberOfInputNeurons];
            weighedSumsHiddenLayer[0] = new double[numberOfHiddenNeurons[0]];
            hiddenLayerOutputs[0] = new double[numberOfHiddenNeurons[0]];
            hiddenLayerBiasDelta[0] = new double[numberOfHiddenNeurons[0]];
            hiddenLayerFactors[0] = new double[numberOfHiddenNeurons[0]];

            for (int i = 1; i < numberOfHiddenLayers; i++)
            {
                hiddenLayerMatrix[i] = new double[numberOfHiddenNeurons[i], numberOfHiddenNeurons[i-1]];
                hiddenLayerGradientMatrix[i] = new double[numberOfHiddenNeurons[i], numberOfHiddenNeurons[i - 1]];
                hiddenLayerDeltaMatrix[i] = new double[numberOfHiddenNeurons[i], numberOfHiddenNeurons[i - 1]];
                hiddenLayerBias[i] = new double[numberOfHiddenNeurons[i]];
                hiddenLayerBiasGradient[i] = new double[numberOfHiddenNeurons[i]];
                weighedSumsHiddenLayer[i] = new double[numberOfHiddenNeurons[i]];
                hiddenLayerOutputs[i] = new double[numberOfHiddenNeurons[i]];
                hiddenLayerBiasDelta[i] = new double[numberOfHiddenNeurons[i]];
                hiddenLayerFactors[i] = new double[numberOfHiddenNeurons[i]];
            }
            

            // hiddenLayerBias[i] = new double[numberOfHiddenNeurons[i]];
            for (int i = 0; i < numberOfHiddenLayers; i++)
            {
                for(int j = 0; j < numberOfHiddenNeurons[i]; j++)
                {
                    int numberOfPreviousLayerOutputs;
                    if (i == 0)
                    {
                        numberOfPreviousLayerOutputs = numberOfInputNeurons;
                    }
                    else
                    {
                        numberOfPreviousLayerOutputs = numberOfHiddenNeurons[i - 1];
                    }
                    for (int k = 0; k < numberOfPreviousLayerOutputs; k++)
                    {
                        hiddenLayerMatrix[i][j, k] = 0;
                        hiddenLayerGradientMatrix[i][j,k] = 0;
                        hiddenLayerMatrix[i][j, k] = 0;
                    }
                    hiddenLayerBias[i][j] = 0;
                    hiddenLayerBiasGradient[i][j] = 0;
                    weighedSumsHiddenLayer[i][j] = 0;
                    hiddenLayerOutputs[i][j] = 0;
                    hiddenLayerBiasDelta[i][j] = 0;
                }
            }
            for (int i = 1; i < numberOfHiddenLayers; i++)
            {
                for (int j = 0; j < numberOfHiddenNeurons[i]; j++)
                {
                    for (int k = 0; k < numberOfHiddenNeurons[i-1]; k++)
                    {
                        hiddenLayerMatrix[i][j, k] = 0;
                        hiddenLayerGradientMatrix[i][j, k] = 0;
                        hiddenLayerMatrix[i][j, k] = 0;
                    }
                    hiddenLayerBias[i][j] = 0;
                    hiddenLayerBiasGradient[i][j] = 0;
                    weighedSumsHiddenLayer[i][j] = 0;
                    hiddenLayerOutputs[i][j] = 0;
                    hiddenLayerBiasDelta[i][j] = 0;
                }
            }
            /*
            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                hiddenLayerBiasGradient[i] = 0;
                weighedSumsHiddenLayer[i] = 0;
                hiddenLayerOutputs[i] = 0;
                hiddenLayerBiasDelta[i] = 0;
                for(int j = 0; j< numberOfInputNeurons; j++)
                {
                    hiddenLayerGradientMatrix[i,j] = 0;
                    hiddenLayerDeltaMatrix[i,j] = 0;
                }
            }*/

            // uwaga na ta linijke
            outputLayerMatrix = new double[numberOfOutputs, numberOfHiddenNeurons[numberOfHiddenLayers - 1]];
            outputLayerBias = new double[numberOfOutputs];
            outputLayerBiasGradient = new double[numberOfOutputNeurons];
            outputLayerFactor = new double[numberOfOutputNeurons];
            outputLayerOutputs = new double[numberOfOutputNeurons];

            weighedSumsOutputLayer = new double[numberOfOutputNeurons];  //tu bylo numberOfHiddenNeurons, ale to byl razcej blad (nie krytyczny, program dzialal, ale duzy blad)
            // idk czy to jest dobrze
            outputLayerGradientMatrix = new double[numberOfOutputNeurons, numberOfHiddenNeurons[numberOfHiddenNeurons.Count-1]];
            outputLayerDeltaMatrix = new double[numberOfOutputNeurons, numberOfHiddenNeurons[numberOfHiddenNeurons.Count - 1]];
            outputLayerBiasDelta = new double[numberOfOutputNeurons];
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                outputLayerBiasGradient[i] = 0;
                weighedSumsOutputLayer[i] = 0;
                outputLayerOutputs[i] = 0;
                outputLayerBiasDelta[i] = 0;
                for(int j = 0; j< numberOfHiddenNeurons[numberOfHiddenNeurons.Count - 1]; j++)
                {
                    outputLayerGradientMatrix[i,j] = 0;
                    outputLayerDeltaMatrix[i, j] = 0;
                }
            }


            data = new List<List<double>>();
            desiredOutputs = new List<List<double>>();
        }
        public void LoadData(string filename)
        {
            string[] lines = File.ReadAllLines(filename);
            foreach (string line in lines)
            {
                string[] columns = line.Split(',');
                List<double> rowData = new List<double>(columns.Length);

                // Parse the first four columns as double values
                for (int i = 0; i < 4; i++)
                {
                    rowData.Add(ConvertToDouble(columns[i]));
                }
                if (columns[4] == "Iris-setosa")
                {
                    List<double> output = new List<double>();
                    rowData.Add(1);
                    rowData.Add(0);
                    rowData.Add(0);
                }
                else if (columns[4] == "Iris-versicolor")
                {
                    List<double> output = new List<double>();
                    rowData.Add(0);
                    rowData.Add(1);
                    rowData.Add(0);
                    desiredOutputs.Add(output);
                }
                else if (columns[4] == "Iris-virginica")
                {
                    List<double> output = new List<double>();
                    rowData.Add(0);
                    rowData.Add(0);
                    rowData.Add(1);
                    desiredOutputs.Add(output);
                }
                data.Add(rowData);
            }
        }
        public static double ConvertToDouble(string input)
        {
            double result = 0.0;
            double sign = 1.0;
            double fraction = 0.0;
            bool hasFraction = false;
            int decimalPlaces = 1;
            int currentIndex = 0;

            // Check if the input string is empty
            if (string.IsNullOrEmpty(input))
            {
                throw new ArgumentException("Invalid input. The string is empty.");
            }

            // Check for a sign character (+ or -) at the beginning
            if (input[currentIndex] == '-')
            {
                sign = -1.0;
                currentIndex++;
            }
            else if (input[currentIndex] == '+')
            {
                currentIndex++;
            }

            // Iterate over the remaining characters
            while (currentIndex < input.Length)
            {
                char currentChar = input[currentIndex];

                // Check for a decimal point
                if (currentChar == '.')
                {
                    if (hasFraction)
                    {
                        throw new ArgumentException("Invalid input. The string contains multiple decimal points.");
                    }

                    hasFraction = true;
                    currentIndex++;
                    continue;
                }

                // Convert the character to a digit
                int digit = currentChar - '0';

                // Check if the character is a valid digit
                if (digit < 0 || digit > 9)
                {
                    throw new ArgumentException("Invalid input. The string contains non-numeric characters.");
                }

                // Update the result based on the current digit
                if (hasFraction)
                {
                    fraction = fraction * 10.0 + digit;
                    decimalPlaces *= 10;
                }
                else
                {
                    result = result * 10.0 + digit;
                }

                currentIndex++;
            }

            // Apply the sign and fraction (if any) to the final result
            result = sign * (result + fraction / decimalPlaces);

            return result;
        }

        public void Train(int numberOfEpochs, double learningRate,double momentum,bool bias,bool shuffle)
        {
            //NA RAZIE JEST HARDCODOWANE BO CHCE MIEC TYLKO ROZPISANY ALGORYTM (numberOfSamples ofc)
            int numberOfSamples = 130;
            InitializeWeightsAndBiases(bias);

            for(int i =0; i<numberOfEpochs; i++)
            {
                double mse = 0d;
                if (shuffle)
                {
                    ShuffleData();
                }
                for(int j = 0; j < numberOfSamples; j++)
                {
                    getSample(j);
                    ForwardPropagation();
                    mse+=CalculateCostFunction();
                    BackwardPropagation();
                    ApplyNewWeights(learningRate,momentum);
                }
                //Console.WriteLine("Blad dla calej warstwy:" + mse);
                //Console.WriteLine("Sredni blad dla calej warstwy:" + mse/(double)numberOfSamples);
            }
        }

        public void SaveNetworkToFile(String filename)
        {
            try
            {
                // Create a FileStream to write the serialized data
                FileStream fileStream = new FileStream(filename, FileMode.Create);

                // Create a BinaryFormatter to perform the serialization
                BinaryFormatter formatter = new BinaryFormatter();

                // Serialize the object to the file
                formatter.Serialize(fileStream, this);

                // Close the file stream
                fileStream.Close();

                Console.WriteLine("Class saved to file: " + filename);
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error saving class to file: " + ex.Message);
            }
        }
        
        public static MLP ReadNetworkFromFile(String filename)
        {
            try
            {
                // Create a FileStream to read the serialized data
                FileStream fileStream = new FileStream(filename, FileMode.Open);

                // Create a BinaryFormatter to perform the deserialization
                BinaryFormatter formatter = new BinaryFormatter();

                // Deserialize the object from the file
                MLP obj = (MLP)formatter.Deserialize(fileStream);

                // Close the file stream
                fileStream.Close();

                Console.WriteLine("Class loaded from file: " + filename);

                return obj;
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error reading class from file: " + ex.Message);
                return null;
            }
        }

        private void ShuffleData()
        {
            Random random = new Random();

            // Start from the end and swap each element with a randomly selected one
            for (int i = data.Count - 1; i > 0; i--)
            {
                int j = random.Next(i + 1);
                List<double> temp = data[i];
                data[i] = data[j];
                data[j] = temp;
            }
        }

        public void Test(int firstSampleIndex, int lastSampleIndex)
        {
            Console.WriteLine("The result:");
            int correctlyClassified = 0;
            int incorrectlyClassified = 0;
            for(int i = firstSampleIndex; i < lastSampleIndex; i++)
            {
                getSample(i);
                ForwardPropagation();
                double cost = CalculateCostFunction();
                if (cost < 0.005)
                {
                    correctlyClassified++;
                }
                else
                {
                    incorrectlyClassified++;
                }
            }
            Console.WriteLine("Correctly classified:" + correctlyClassified);
            Console.WriteLine("Incorrectly classified:" + incorrectlyClassified);
        }


        private void getSample(int numberOfSample)
        {
            for(int i = 0; i < numberOfInputNeurons; i++)
            {
                currentSample[i] = data.ElementAt(numberOfSample).ElementAt(i);
            }
            for(int i = numberOfInputNeurons; i< numberOfOutputNeurons+numberOfInputNeurons; i++)
            {
                desiredOutput[i-numberOfInputNeurons] = data.ElementAt(numberOfSample).ElementAt(i);
            }
        }
        private void ApplyNewWeights(double learningRate,double momentum)
        {
            //Applying for hidden layers
            for(int layerIndex = 0; layerIndex < numberOfHiddenLayers;layerIndex++)
            {
                for (int i = 0; i < numberOfHiddenNeurons[layerIndex]; i++)
                {
                    int numberOfInputNeuronsIntoLayer;
                    if (layerIndex == 0)
                    {
                        numberOfInputNeuronsIntoLayer = numberOfInputNeurons;
                    }
                    else
                    {
                        numberOfInputNeuronsIntoLayer = numberOfHiddenNeurons[layerIndex - 1];
                    }
                    for (int j = 0; j < numberOfInputNeuronsIntoLayer; j++)
                    {
                        hiddenLayerMatrix[layerIndex][i, j] += learningRate * -1.0 * hiddenLayerGradientMatrix[layerIndex][i, j] 
                            + hiddenLayerDeltaMatrix[layerIndex][i,j]*momentum; 
                        // MOMENTUM
                        hiddenLayerDeltaMatrix[layerIndex][i,j] = learningRate * -1.0 * hiddenLayerGradientMatrix[layerIndex][i, j]
                            + hiddenLayerDeltaMatrix[layerIndex][i, j] * momentum;
                    }
                    hiddenLayerBias[layerIndex][i] += learningRate * -1.0 * hiddenLayerBiasGradient[layerIndex][i] + hiddenLayerBiasDelta[layerIndex][i] * momentum;
                    hiddenLayerBiasDelta[layerIndex][i] = learningRate * -1.0 * hiddenLayerBiasGradient[layerIndex][i] + hiddenLayerBiasDelta[layerIndex][i] * momentum;
                }   
            }
            //Apply for output layer
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                for (int j = 0; j < numberOfHiddenNeurons[numberOfHiddenLayers-1]; j++)
                {
                    outputLayerMatrix[i, j] += learningRate * -outputLayerGradientMatrix[i, j] + outputLayerDeltaMatrix[i,j]*momentum;
                    // MOMENTUM
                    outputLayerDeltaMatrix[i,j] = learningRate * -outputLayerGradientMatrix[i, j] + outputLayerDeltaMatrix[i, j] * momentum;
                }
                
                outputLayerBias[i] += learningRate * -1.0 * outputLayerBiasGradient[i] + outputLayerBiasDelta[i]*momentum;
                outputLayerBiasDelta[i] = learningRate * -1.0 * outputLayerBiasGradient[i] + outputLayerBiasDelta[i]*momentum;
                
            }
        }

        private void InitializeWeightsAndBiases(bool bias)
        {
            Random random = new Random();

            // INITIALIZATION FOR HIDDEN LAYER
            for(int layerIndex = 0; layerIndex < numberOfHiddenLayers; layerIndex++)
            {
                for ( int i = 0; i < numberOfHiddenNeurons[layerIndex] ; i++)
                {
                    int numberOfInputNeuronsIntoLayer;
                    if (layerIndex == 0)
                    {
                        numberOfInputNeuronsIntoLayer = numberOfInputNeurons;
                    }
                    else
                    {
                        numberOfInputNeuronsIntoLayer = numberOfHiddenNeurons[layerIndex - 1];
                    }
                    for (int j = 0; j < numberOfInputNeuronsIntoLayer; j++)
                    { 
                        hiddenLayerMatrix[layerIndex][i,j] = random.NextDouble() * 2 - 1; // randomizing a number in range [-1,1]
                    }
                    if (bias)
                    {
                        hiddenLayerBias[layerIndex][i] = random.NextDouble() * 2 - 1;   // same here
                    }
                    else
                    {
                        hiddenLayerBias[layerIndex][i] = 0d;
                    }
                }
            }
            // INITIALIZATION FOR OUTPUT LAYER
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                for (int j = 0; j < numberOfHiddenNeurons[numberOfHiddenLayers-1]; j++)
                {
                    outputLayerMatrix[i, j] = random.NextDouble() * 2 - 1; // randomizing a number in range [-1,1]
                }
                if (bias)
                {
                    outputLayerBias[i] = random.NextDouble() * 2 - 1; // same here
                }
                else
                {
                    outputLayerBias[i] = 0;
                }
            }

        }
        private void ForwardPropagation()
        {
            // FORWARD PROPGATION FOR HIDDEN LAYER
            
            for(int layerIndex = 0; layerIndex< numberOfHiddenLayers; layerIndex++)
            {
                for(int i = 0; i<numberOfHiddenNeurons[layerIndex]; i++)
                {
                    weighedSumsHiddenLayer[layerIndex][i] = 0;
                    int numberOfInputNeuronsIntoLayer;
                    if (layerIndex == 0)
                    {
                        numberOfInputNeuronsIntoLayer = numberOfInputNeurons;
                    }
                    else
                    {
                        numberOfInputNeuronsIntoLayer = numberOfHiddenNeurons[layerIndex - 1];
                    }
                    for (int j = 0; j < numberOfInputNeuronsIntoLayer; j++)
                    {
                        if (layerIndex == 0)
                        {
                            weighedSumsHiddenLayer[layerIndex][i] += currentSample[j] * hiddenLayerMatrix[layerIndex][i,j]; 
                        }
                        else
                        {
                            weighedSumsHiddenLayer[layerIndex][i] += hiddenLayerOutputs[layerIndex-1][j] * hiddenLayerMatrix[layerIndex][i, j];
                        }
                    }
                    weighedSumsHiddenLayer[layerIndex][i] += hiddenLayerBias[layerIndex][i];
                    // not sure if it's the right place for this line but whatever, I guess it is
                    hiddenLayerOutputs[layerIndex][i] = Sigmoid(weighedSumsHiddenLayer[layerIndex][i]);
                }
            }

            // FORWARD PROPGATION FOR OUTPUT LAYER
            
            for (int i = 0; i<numberOfOutputNeurons; i++)
            {
                weighedSumsOutputLayer[i] = 0;
                for(int j = 0; j< numberOfHiddenNeurons[numberOfHiddenLayers-1]; j++)
                {
                    weighedSumsOutputLayer[i] += hiddenLayerOutputs[numberOfHiddenLayers - 1][j] * outputLayerMatrix[i, j];
                }
                weighedSumsOutputLayer[i] += outputLayerBias[i];
                // there was another loop that was iterating over the same range, so i moved the line below here //DELETE THE COMMENT LATER
                outputLayerOutputs[i] = Sigmoid(weighedSumsOutputLayer[i]);
            }
        }

        private double CalculateCostFunction()
        {
            double costFunction = 0;
            for(int i = 0; i < numberOfOutputNeurons ; i++)
            {
                costFunction += 0.5 * Math.Pow((outputLayerOutputs[i] - desiredOutput[i]),2);
            }
            costFunction /= numberOfOutputNeurons;
            return costFunction;
        }

        private void BackwardPropagation()
        {
            // OUTPUT LAYER BACKWARD PROPAGATION

            for(int i = 0; i < numberOfOutputNeurons ; i++)
            {
                outputLayerFactor[i] = SigmoidDerivative(weighedSumsOutputLayer[i]) * (outputLayerOutputs[i] - desiredOutput[i]);
                for(int j = 0; j<numberOfHiddenNeurons[numberOfHiddenLayers - 1] ; j++)
                {
                    // formula is on paper, point 8
                    outputLayerGradientMatrix[i, j] = hiddenLayerOutputs[numberOfHiddenLayers - 1][j] * outputLayerFactor[i];
                }
                // The weight of the bias term in a layer is updated in the same fashion as all the other weights are.
                // What makes it different is that it is independent of output from previous layers.
                // The weight for the bias term in a layer is always fed an input of 1.
                // src: https://www.quora.com/How-is-bias-updated-in-neural-network
                // PSA: Yes I am really desperate so I'm gonna take answers from quora.
                // Take this with a pinch of salt, I am not sure if I'm doing it the right way.
                outputLayerBiasGradient[i] = outputLayerFactor[i];
            }


            // HIDDEN LAYERS BACKWARD PROPAGATION
            for(int layerIndex = numberOfHiddenLayers - 1; layerIndex >=0; layerIndex--)
            {
                for (int i = 0; i < numberOfHiddenNeurons[layerIndex]; i++)
                {
                    //CALCULATING NEURON FACTOR
                    // FOR LAST HIDDEN LAYER
                    hiddenLayerFactors[layerIndex][i] = 0;
                    if (layerIndex == numberOfHiddenLayers - 1)
                    {
                        for (int outputIndex = 0; outputIndex < numberOfOutputNeurons; outputIndex++)
                        {
                            hiddenLayerFactors[layerIndex][i] += outputLayerFactor[outputIndex] * outputLayerMatrix[outputIndex, i];
                        }
                        hiddenLayerFactors[layerIndex][i] *= SigmoidDerivative(hiddenLayerOutputs[layerIndex][i]);
                        hiddenLayerBiasGradient[layerIndex][i] = hiddenLayerFactors[layerIndex][i];
                    }
                    else
                    {
                        int numberOfNeuronsInNextLayer = numberOfHiddenNeurons[layerIndex+1];
                        for (int outputIndex = 0; outputIndex < numberOfNeuronsInNextLayer; outputIndex++)
                        {
                            hiddenLayerFactors[layerIndex][i] += hiddenLayerFactors[layerIndex+1][outputIndex] * hiddenLayerMatrix[layerIndex+1][outputIndex, i];
                        }
                        hiddenLayerFactors[layerIndex][i] *= SigmoidDerivative(hiddenLayerOutputs[layerIndex][i]);
                        hiddenLayerBiasGradient[layerIndex][i] = hiddenLayerFactors[layerIndex][i];
                    }

                    int numberOfLayerInputs;
                    if(layerIndex == numberOfHiddenLayers-1)
                    {
                        numberOfLayerInputs = numberOfHiddenNeurons[layerIndex - 1];
                        for (int j = 0; j < numberOfLayerInputs; j++)
                        {
                            // should we divide it by the number of output neurons or not?
                            hiddenLayerGradientMatrix[layerIndex][i, j] = 1.0 / numberOfOutputNeurons * hiddenLayerFactors[layerIndex][i] * hiddenLayerOutputs[layerIndex-1][j];
                        }
                    }
                    else if(layerIndex == 0)
                    {
                        numberOfLayerInputs = numberOfInputNeurons;
                        for (int j = 0; j < numberOfLayerInputs; j++)
                        {
                            hiddenLayerGradientMatrix[layerIndex][i, j] = 1.0 / numberOfHiddenNeurons[layerIndex + 1] * hiddenLayerFactors[layerIndex][i] * currentSample[j];
                        }
                    }
                    else
                    {
                        numberOfLayerInputs = numberOfHiddenNeurons[layerIndex - 1];
                        for (int j = 0; j < numberOfLayerInputs; j++)
                        {
                            hiddenLayerGradientMatrix[layerIndex][i, j] = 1.0 / numberOfHiddenNeurons[layerIndex + 1] * hiddenLayerFactors[layerIndex][i] * hiddenLayerOutputs[layerIndex - 1][j];
                        }
                    }
                    

                    /*
                    int numberOfInputNeuronsIntoLayer;
                    if (layerIndex == 0)
                    {
                        numberOfInputNeuronsIntoLayer = numberOfInputNeurons;
                    }
                    else
                    {
                        numberOfInputNeuronsIntoLayer = numberOfHiddenNeurons[layerIndex - 1];
                    }
                    for (int j = 0; j < numberOfInputNeuronsIntoLayer; j++)
                    {
                        // I added this not so long ago, I didn't clear out the matrix.
                        hiddenLayerGradientMatrix[layerIndex][i, j] = 0;
                        for (int k = 0; k < numberOfOutputNeurons; k++)
                        {
                            // formula is on paper, point 9
                            // I swear to god if this works i'm going to apply to openai
                            if(layerIndex == numberOfHiddenLayers - 1)
                            {
                                hiddenLayerGradientMatrix[layerIndex][i, j] += outputLayerFactor[k] * outputLayerMatrix[k, i];
                            }
                            else if(layerIndex == 0)
                            {

                            }
                            else
                            {
                                hiddenLayerGradientMatrix[layerIndex][i, j] += (outputLayerOutputs[k] - desiredOutput[k]) * SigmoidDerivative(weighedSumsHiddenLayer[layerIndex+1][k]) * outputLayerMatrix[k, i] *
                                    SigmoidDerivative(weighedSumsHiddenLayer[layerIndex][i]) * currentSample[j];
                            }
                        }
                        double result = hiddenLayerGradientMatrix[layerIndex][i, j] * (1.0 / numberOfOutputNeurons);
                        hiddenLayerGradientMatrix[layerIndex][i, j] = result;
                    }
                    */

                }
                /*
                // HERE WE CALCULATE BIAS DERIVATIVE FOR HIDDEN LAYER
                // Take this with a pinch of salt, I am not sure if I'm doing it the right way.
                for(int i = 0; i < numberOfHiddenNeurons; i++)
                {
                    hiddenLayerBiasGradient[i] = 0;
                    for(int j = 0; j< numberOfOutputNeurons; j++)
                    {
                        hiddenLayerBiasGradient[i] += (outputLayerOutputs[j] - desiredOutput[j]) * SigmoidDerivative(weighedSumsOutputLayer[j]) * SigmoidDerivative(weighedSumsHiddenLayer[i]) *
                            outputLayerMatrix[j, i];
                    }
                    hiddenLayerBiasGradient[i] *= 1.0 / numberOfOutputNeurons;
                }
                */
            }
        }


        private double Sigmoid(double x)
        {
            return 1.0 / (1.0 + Math.Exp(-x));
        }

        private double SigmoidDerivative(double x)
        {
            return Sigmoid(x)*(1-Sigmoid(x));
        }
    }
}
