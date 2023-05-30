
using MLP_TAKE2;
using System.Runtime.Serialization;
using System.Runtime.Serialization.Formatters.Binary;

/*
MLP mlp = new MLP(4, 4, 3);

mlp.LoadIrisData("../../../Data/training.data");

Console.WriteLine("TRAINING:");
mlp.Train(1000,0.01,0.9, true,true,0.01);
Console.WriteLine("FINISHED TRAINING:");
Console.WriteLine("");
Console.WriteLine("Testing learning samples:");
mlp.TestIrises();


Stream str = new FileStream("../../../Networks/network.file", FileMode.Create, FileAccess.Write);
IFormatter f = new BinaryFormatter();
f.Serialize(str, mlp);
str.Close();
str = new FileStream("../../../Networks/network.file", FileMode.Open, FileAccess.Read);
MLP mlp2 = (MLP)f.Deserialize(str);

Console.WriteLine("Tests:");
mlp2.LoadIrisData("../../../testing.data");
mlp2.TestIrises();

*/
/*
IFormatter f = new BinaryFormatter();


//MLP mlp = new MLP(4, 2, 4);
mlp.LoadAutoEncoderData("../../../Data/autoencoder.data");
mlp.InitializeWeightsAndBiases(true);

Stream str = new FileStream("../../../Networks/network.file", FileMode.Create, FileAccess.Write);

f.Serialize(str, mlp);
str.Close();


Stream str2 = new FileStream("../../../Networks/network.file", FileMode.Open, FileAccess.Read);
*/

/*
MLP mlp2 = new MLP(4, 2, 4);
mlp2.LoadAutoEncoderData("../../../Data/autoencoder.data");
//mlp2.SetBiasesToZero();
mlp2.Train(2000, 0.7, 0.6, true, true, 0.001, false);
mlp2.TestIrises();
*/


MLP mlp = new MLP(4, 3, 3);
mlp.LoadIrisData("../../../Data/training.data");
mlp.Train(2000, 0.7, 0, true, true, 1.0, false);
mlp.LoadIrisData("../../../Data/testing.data");
mlp.TestIrises();

