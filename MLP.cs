using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
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
        private double[,] hiddenLayerMatrix;
        private double[,] outputLayerMatrix;

        private double[,] hiddenLayerGradientMatrix;
        private double[,] outputLayerGradientMatrix;


        private double[] hiddenLayerBias;
        private double[] outputLayerBias;


        private int numberOfInputNeurons;
        private int numberOfOutputNeurons;
        private int numberOfHiddenNeurons;


        private double[] weighedSumsHiddenLayer;
        private double[] hiddenLayerOutputs;

        private double[] weighedSumsOutputLayer;
        private double[] outputLayerOutputs;




        private List<List<double>> data;
        private List<List<double>> desiredOutputs;

        public MLP(int numberOfInputNeurons, int numberOfHiddenNeurons, int numberOfOutputs, bool bias)
        {
            this.numberOfInputNeurons = numberOfInputNeurons;
            this.numberOfHiddenNeurons = numberOfHiddenNeurons;
            this.numberOfOutputNeurons = numberOfOutputs;

            desiredOutput = new double[numberOfOutputs];
            currentSample = new double[numberOfInputNeurons];
            hiddenLayerMatrix = new double[numberOfHiddenNeurons, numberOfInputNeurons];
            outputLayerMatrix = new double[numberOfOutputs, numberOfHiddenNeurons];
            hiddenLayerBias = new double[numberOfHiddenNeurons];
            outputLayerBias = new double[numberOfOutputs];


            weighedSumsHiddenLayer = new double[numberOfHiddenNeurons];
            hiddenLayerOutputs = new double[numberOfHiddenNeurons];
            hiddenLayerGradientMatrix = new double[numberOfHiddenNeurons, numberOfInputNeurons];
            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                weighedSumsHiddenLayer[i] = 0;
                hiddenLayerOutputs[i] = 0;
                for(int j = 0; j< numberOfInputNeurons; j++)
                {
                    hiddenLayerGradientMatrix[i,j] = 0;
                }
            }
            
            outputLayerOutputs = new double[numberOfOutputNeurons];
            weighedSumsOutputLayer = new double[numberOfOutputNeurons]; //tu bylo numberOfHiddenNeurons, ale to byl razcej blad (nie krytyczny, program dzialal, ale duzy blad)
            outputLayerGradientMatrix = new double[numberOfOutputNeurons, numberOfHiddenNeurons];
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                weighedSumsOutputLayer[i] = 0;
                outputLayerOutputs[i] = 0;
                for(int j = 0; j< numberOfHiddenNeurons; j++)
                {
                    outputLayerGradientMatrix[i,j] = 0;
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
                    double test;
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
        private void LoadMockData(int index)
        {
            if(index == 0)
            {
                currentSample[0] = 5.1;
                currentSample[1] = 3.5;
                currentSample[2] = 1.4;
                currentSample[3] = 0.2;

                desiredOutput[0] = 1;
                desiredOutput[1] = 0;
                desiredOutput[2] = 0;
            }

        }

        public void Train(int numberOfEpochs, double learningRate,double momentum)
        {
            //NA RAZIE JEST HARDCODOWANE BO CHCE MIEC TYLKO ROZPISANY ALGORYTM (numberOfSamples ofc)
            int numberOfSamples = 75;
            InitializeWeightsAndBiases();

            for(int i =0; i<numberOfEpochs; i++)
            {
                ShuffleData();
                //Console.WriteLine("Epoch " + i.ToString() + ":");
                for(int j = 0; j < numberOfSamples; j++)
                {
                    getSample(j);
                    ForwardPropagation();
                    CalculateCostFunction();
                    BackwardPropagation();
                    ApplyNewWeights(learningRate);
                }
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
                if (cost < 0.05)
                {
                    correctlyClassified++;
                }
                else
                {
                    incorrectlyClassified++;
                }
                /*
                Console.Write(outputLayerOutputs[0]);
                Console.Write(outputLayerOutputs[1]);
                Console.Write(outputLayerOutputs[2]);
                */
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
        private void ApplyNewWeights(double learningRate)
        {
            //Applying for hidden layer
            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    hiddenLayerMatrix[i, j] += learningRate * -hiddenLayerGradientMatrix[i, j];
                }
            }
            //Apply for output layer
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                for (int j = 0; j < numberOfHiddenNeurons; j++)
                {
                    outputLayerMatrix[i, j] = learningRate * -outputLayerGradientMatrix[i, j]; // randomizing a number in range [-1,1]
                }
            }
        }

        private void InitializeWeightsAndBiases()
        {
            Random random = new Random();

            // INITIALIZATION FOR HIDDEN LAYER
            for ( int i = 0; i < numberOfHiddenNeurons ; i++)
            {
                for(int j = 0; j < numberOfInputNeurons ; j++)
                { 
                    hiddenLayerMatrix[i,j] = random.NextDouble() * 2 - 1; // randomizing a number in range [-1,1]
                }
                hiddenLayerBias[i] = random.NextDouble() * 2 - 1;   // same here
            }
            // INITIALIZATION FOR OUTPUT LAYER
            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
                for (int j = 0; j < numberOfHiddenNeurons; j++)
                {
                    outputLayerMatrix[i, j] = random.NextDouble() * 2 - 1; // randomizing a number in range [-1,1]
                }
                outputLayerBias[i] = random.NextDouble() * 2 - 1; // same here
            }

        }
        private void ForwardPropagation()
        {
            // FORWARD PROPGATION FOR HIDDEN LAYER
            

            for(int i = 0; i<numberOfHiddenNeurons; i++)
            { 
                for(int j = 0; j < numberOfInputNeurons; j++)
                {
                    weighedSumsHiddenLayer[i] += currentSample[j] * hiddenLayerMatrix[i,j]; 
                }
                weighedSumsHiddenLayer[i] += hiddenLayerBias[i];
            }
            

            for(int i = 0; i < numberOfHiddenNeurons; i++)
            {
                hiddenLayerOutputs[i] = Sigmoid(weighedSumsHiddenLayer[i]);
            }

            // FORWARD PROPGATION FOR OUTPUT LAYER
            
            for (int i = 0; i<numberOfOutputNeurons; i++)
            {
                for(int j = 0; j< numberOfHiddenNeurons; j++)
                {
                    weighedSumsOutputLayer[i] += hiddenLayerOutputs[j] * outputLayerMatrix[i, j];
                }
                weighedSumsOutputLayer[i] += outputLayerBias[i];
            }

            for (int i = 0; i < numberOfOutputNeurons; i++)
            {
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
                for(int j = 0; j<numberOfHiddenNeurons ; j++)
                {
                    // formula is on paper, point 8
                    outputLayerGradientMatrix[i, j] = hiddenLayerOutputs[j] * SigmoidDerivative(weighedSumsOutputLayer[i])
                        * (outputLayerOutputs[i] - desiredOutput[i]);
                }
            }


            // HIDDEN LAYER BACKWARD PROPAGATION

            for (int i = 0; i < numberOfHiddenNeurons; i++)
            {
                for (int j = 0; j < numberOfInputNeurons; j++)
                {
                    for (int k = 0; k < numberOfOutputNeurons; k++)
                    {
                        // formula is on paper, point 9
                        // I swear to god if this works i'm going to apply to openai
                        hiddenLayerGradientMatrix[i, j] += (outputLayerOutputs[k] - desiredOutput[k]) * SigmoidDerivative(weighedSumsOutputLayer[k]) * outputLayerMatrix[k, i] *
                            SigmoidDerivative(weighedSumsHiddenLayer[i]) * currentSample[j];
                    }
                    double result = hiddenLayerGradientMatrix[i, j] * (1.0 / numberOfOutputNeurons);
                    hiddenLayerGradientMatrix[i, j] = result;
                }
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
