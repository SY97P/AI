package 과제;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;

public class MLP {
	
	/*
	global variable list
	input pattern size /, hidden node number /, learning rate /, 
	training data number /, test data number /, training repeat number /...
	*/
	
	// loss rate이 5% 미만으로 나와야 함
	
	// Model Structure
	static int input_size = 784;		// input layer node 개수
	static int hidden_size = 100;		// hidden layer node 개수
	static int output_size = 10;		// output layer node 개수
	
	static double learningRate = 0.1;	// learning rate, 에타는 0.1 ~ 0.5
	
	static int train_data_num = 49000;
	static int test_data_num = 21000;
	static int EPOCH = 200;			// 반복은 100번
	
	
	public static void main(String[] args) throws FileNotFoundException {
		// TODO 자동 생성된 메소드 스텁
		
		// file call
		File file = new File("C:\\Users\\SYP\\Desktop\\MNIST.txt");
		
		
		// initialize variable(hidden layer weight and threshold,
		// output layer weight and threshold, data array...)
		// 1 step : initialize method call (hidden & output layer)
		double hidden_weight[][] = new double[100][784];	// row : hidden_sise, col = output_size
		double hidden_threshold[] = new double[100];		// hidden node 개수 100개 = threshold 개수 100개
		double output_weight[][] = new double[10][100]; 	// row : output 개수, col = input 개수
		double output_threshold[] = new double[10];			// output node 개수와 동일
		

		initialize(hidden_threshold, hidden_weight);
		initialize(output_threshold, output_weight);
		
		Scanner pixel = new Scanner(file);
		// training sample 반복 횟수
		int epoch = 0;
		// 지금이 몇 번째 training sample인지
		// int trainig_index = 0;

		while(pixel.hasNext()) {
			/*
			 * initialize variable (input array, hidden layer value array,
			 * output array, output layer value array, error gradient array ...)
			 * 
			 * 다시 말해서 원하는 정답과 함께 기록된 하나의 training data를 단위로 while문에 들어올 거임
			 * while문에 들어올 때마다, 한번의 training을 한 것.
			 */
			// 맨 처음 file에서 들어가는 input (픽셀 값 : 0~255)
			double input_arr[] = new double[input_size];
			// 예측 value가 1.0이고 나머지는 0.0인 output (0~9까지의 확률)
			double output_arr[] = new double[output_size];
			// hidden layer node의 값 : 나중에 input에서의 activation func 결과가 들어감
			// 각 class에 대한 확률이 나올 것으로 예상됨.
			double hidden_layer_arr[] = new double[hidden_size];
			// ouput layer node의 값 : hidden layer를 input으로 한 activation func 결과가 들어감
			// 0~9 까지의 각 숫자에 대한 확률이 나올 것으로 예상됨
			double output_layer_arr[] = new double[output_size];
			// error gradient는 weight 값이 변함에 따라 error 가 변하는 비율을 말함
			double err_grad_arr[] = new double[output_size];
			
			float loss_rate = 0;
			float loss_count = 0;

			// 한 번의 반복마다 순차적으로 sample을 바꾸고 싶음
			for (int curr = 0; curr < train_data_num; curr++) {
				// One-hot encoding for output
				// 해당 training sample의 예상 value 값을 output_arr에 저장
				int index = pixel.nextInt();
				for (int i = 0; i < output_arr.length; i++)
					output_arr[i] = 0.0;
				output_arr[index] = 1.0;
				// 총 784개의 픽셀을 input_arr에 저장
				// 한 sample 을 input_arr[]에 저장
				for (int i = 0; i < input_size; i++) {
					input_arr[i] = (double) pixel.nextInt();
				}

				// 2 step : activation method call (hidden & output layer)
				// 이로써 input에서 hidden_layer의 node 값을 구함
				activation(input_arr, hidden_weight, hidden_threshold, hidden_layer_arr);
				// 이로써 hidden_layer에서 output_layer의 node 값을 구함
				activation(hidden_layer_arr, output_weight, output_threshold, output_layer_arr);

				
				for(int i=0; i< output_size; i++)
					 System.out.print(output_layer_arr[i]+" ");
				System.out.println();
				 

				// 3 step : training weight value
				// 3-1 : output 오차 기울기 구하기
				for (int i = 0; i < output_size; i++) {
					err_grad_arr[i] = output_arr[i] * (1 - output_arr[i]) * (output_layer_arr[i] - output_arr[i]);
					// 3-2 : output weight 업데이트하기
					for (int j = 0; j < hidden_size; j++) {
						// output_weight[i][j] += learningRate * err_grad_arr[i] * hidden_layer_arr[j];
						output_weight[i][j] += learningRate * err_grad_arr[i] * output_layer_arr[i];
						// hidden_layer_arr을 output_layer_arr로 바꾸어보자
					}
				}
				// 3-3 : hidden 오차 기울기 구하기 = value*(1-value)*sum(outputlayer 오차기울기 * weight)
				for (int i = 0; i < output_size; i++) {
					double sum = output_layer_arr[i] * (1 - output_layer_arr[i]);
					for (int j = 0; j < hidden_size; j++) {
						// sum(outputlayer 오차 기울기 * hiddenTooutput weight)
						sum = err_grad_arr[i] * output_weight[i][j];
					}
					// sum은 구했고 hidden 오차 기울기는 못 구한 상태

					// 3-4 : hidden weight 업데이트하기
					//for (int j = 0; j < hidden_size; j++) {
					//	sum = sum * hidden_layer_arr[j] * (1 - hidden_layer_arr[j]);
						
						// sum = sum * output_layer_arr * (1- output_layer_arr)로 수정해보기
					//	for (int k = 0; k < input_size; k++) {
					//		// hidden_weight[j][k] += learningRate * sum * input_arr[k];
					//		hidden_weight[j][k] += learningRate * sum * hidden_layer_arr[j];
					//		// input_arr을 hidden_layer_arr로 수정해보기
					//	}
					//}
					for(int j=0; j< hidden_size; j++) {
						for (int k=0; k< input_size; k++) {
							hidden_weight[j][k] += learningRate * sum * hidden_layer_arr[i];
						}
					}
				}
				// 두 layer의 weight 값을 업데이트 함.
				// 결과적으로 한 번의 while문 당 모든 training data를 모두 적용한 것
				
				/*
				for (int i=0; i< output_size; i++)
					System.out.print(output_layer_arr[i]+" ");
				System.out.println();
				 */
				
				// observation loss rate
				// actual_index 는 output layer node 중 가장 높은 확률의 인덱스 (글자 예상값)
				// target_index 는 output layer node 의 예상했던 값 (한 값만 1.0)
				// 예상한 답이 실제와 맞는지 확인하여 loss rate을 구한다.
				int actual_index = 0;
				double actual_max = 0.0;
				int target_index = 0;
				double target_max = 0.0;
				for (int i = 0; i < output_size; i++) {
					if (actual_max < output_layer_arr[i]) {
						actual_max = output_layer_arr[i];
						actual_index = i;
					}
					if (target_max < output_arr[i]) {
						target_max = output_arr[i];
						target_index = i;
					}
				}
				System.out.println("actual = "+actual_index + " target = "+ target_index);
				// 일단 loss가 난 횟수를 저장
				if (actual_index != target_index)
					loss_count++;
				
				System.out.println("train            = "+ curr);
				System.out.println("train_loss_count = "+loss_count);
				
				// sample 수 조정 (제출시 제거할 것)
				//if (curr > 10000)
				//	break;
			}
			
			// if (all training data is calculated)
			// 방금 for(curr)문으로 모든 training data를 모두 계산함
			// System.out.println("epoch = "+ epoch);
			
			if (epoch < EPOCH) {
				System.out.println();
				//System.out.println("loss_count = "+loss_count);
				//System.out.println(train_data_num);
				// calculate loss rate
				loss_rate = (loss_count / train_data_num) * 100;
				System.out.println("loss_rate = "+ loss_rate);
				// epoch increase
				epoch++;
				System.out.println("epoch = "+ epoch);
				
				// if(filled in the number of training iterations)
				if(epoch >= EPOCH) {
					// start the test
					for (int test = 0; test < test_data_num; test++) {
						int index = pixel.nextInt();
						for (int i = 0; i < output_arr.length; i++)
							output_arr[i] = 0.0;
						output_arr[index] = 1.0;
						for (int i = 0; i < input_size; i++) {
							input_arr[i] = (double) pixel.nextInt();
						}

						activation(input_arr, hidden_weight, hidden_threshold, hidden_layer_arr);
						activation(hidden_layer_arr, output_weight, output_threshold, output_layer_arr);

						int actual_index = 0;
						double actual_max = 0.0;
						int target_index = 0;
						double target_max = 0.0;
						for (int i = 0; i < output_size; i++) {
							if (actual_max < output_layer_arr[i]) {
								actual_max = output_layer_arr[i];
								actual_index = i;
							}
							if (target_max < output_arr[i]) {
								target_max = output_arr[i];
								target_index = i;
							}
						}
						if (actual_index != target_index)
							loss_count++;
						loss_rate = (loss_count / train_data_num) * 100;
						
						System.out.println("test = "+ test);
						
						//if (test > 100)
						//	break;
					}
					System.out.println("test_loss_rate = "+ loss_rate);
					// 나가자
					break;
				}
			}
			//System.out.println();
			pixel = new Scanner(file);
		}
	}
	
	public static void initialize(double threshold[], double weight[][]) {
		/*
		 * Specify the node threshold and weight level as an 
		 * arbitrary number that follows the even distribution
		 */
		
		// repeat
		for (int i=0; i < threshold.length; i++) {
			double thresholdVal = (Math.random() * (4.8/ weight[0].length)) - (2.4 / weight[0].length);
			threshold[i] = thresholdVal;
		}
		for (int i=0; i < weight.length; i++) {
			for (int j=0; j < weight[0].length; j++) {
				// repeat
				double weightVal = (Math.random() * (4.8/weight[0].length)) - (2.4/weight[0].length);
				weight[i][j] = weightVal;
			}
		}
			
	}
	
	public static void activation (double input[], double weight[][], double threshold[], double result[]) {
		/*
		 * i = input 개수(input node, hidden node 상관 없음) = weigth[output][input]
		 */
		for (int i=0; i < result.length; i++) {
			//repeat
			// output에 들어갈 값은 맨 처음 -threshold 값으로 초기화하고 시작
			double resultVal = -threshold[i];
			
			// input * weight의 모든 sum
			for(int j=0; j<input.length; j++) {
				// repeat
				resultVal += input[j] * weight[i][j];
			}
			
			// Sigmoid activation function에 적용하여 output node 값 결정
			result[i] = 1/(1 + Math.exp(-resultVal));
		}
		
	}

}
