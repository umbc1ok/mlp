
using MLP_TAKE2;


MLP mlp = new MLP(4, 4, 3, true);
mlp.LoadData("../../../training.data");

Console.WriteLine("TRAINING:");
mlp.Train(1000,0.1,1);
Console.WriteLine("FINISHED TRAINING:");
Console.WriteLine("Testing learning samples:");
mlp.Test(0, 130);


mlp.LoadData("../../../testing.data");
Console.WriteLine("Tests:");
mlp.Test(0,20);