//使用鼠标在原图像上选取感兴趣区域
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
using namespace std;
using namespace cv;

//计算两点之间的距离
double getDistance(cv::Point2f pointO, cv::Point2f pointA)

{

	double distance;

	distance = powf((pointO.x - pointA.x), 2) + powf((pointO.y - pointA.y), 2);

	distance = sqrtf(distance);



	return distance;

}

cv::Point2f center(0, 0);

cv::Point2f computeIntersect(cv::Vec4i a, cv::Vec4i b)
{
	int x1 = a[0], y1 = a[1], x2 = a[2], y2 = a[3], x3 = b[0], y3 = b[1], x4 = b[2], y4 = b[3];

	if (float d = ((float)(x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)))
	{
		cv::Point2f pt;
		pt.x = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / d;
		pt.y = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / d;
		return pt;
	}
	else
		return cv::Point2f(-1, -1);
}

void sortCorners(std::vector<cv::Point2f>& corners, cv::Point2f center)
{
	std::vector<cv::Point2f> top, bot;

	for (unsigned int i = 0; i< corners.size(); i++)
	{
		if (corners[i].y<center.y)
		{
			top.push_back(corners[i]);
			//cout << "上面每个点" << corners[i];
		}
		else
		{
			bot.push_back(corners[i]);
			//cout << "下面每个点" << corners[i];
		}
	}
	//cout << "大小" << top.size() << endl;

	//cv::Point2f tl, tr, bl, br;
	// tl = top[0].x > top[1].x ? top[1] : top[0];
	// tr = top[0].x > top[1].x ? top[0] : top[1];
	// for (int i = 2; i < top.size(); i++) {
	//	 if (top[i].x < tl.x) {
	//		 tl = top[i];
	//	 }
	//	 else {
	//		 tr = top[i];
	//	 }
	// }

	//for (unsigned int i = 0; i<top.size()-1; i++)
	//{
	//	for (unsigned int j = i + 1; j<top.size(); j++)
	//	{
	//		 tl = top[i].x > top[j].x ? top[j] : top[i];
	//		 tr = top[i].x > top[j].x ? top[i] : top[j];
	//	}
	//}

	//for (unsigned int i = 0; i<bot.size()-1; i++)
	//{
	//	for (unsigned int j = i + 1; j<bot.size(); j++)
	//	{
	//		 bl = bot[i].x > bot[j].x ? bot[j] : bot[i];
	//		 br = bot[i].x > bot[j].x ? bot[i] : bot[j];
	//	}
	//}
	cv::Point2f tl = top[0].x > top[1].x ? top[1] : top[0];
	cv::Point2f tr = top[0].x > top[1].x ? top[0] : top[1];
	cv::Point2f bl = bot[0].x > bot[1].x ? bot[1] : bot[0];
	cv::Point2f br = bot[0].x > bot[1].x ? bot[0] : bot[1];

	corners.clear();
	//注意以下存放顺序是顺时针，当时这里出错了，如果想任意顺序下文开辟的四边形矩阵注意对应
	corners.push_back(tl);
	corners.push_back(tr);
	corners.push_back(br);
	corners.push_back(bl);

}


//void fun();
//void fun(){
//	Mat src_img;
//	src_img = imread("1.jpg");
//	//使用resize()函数统一图像大小
//	Size ResImgSiz = Size(230, 230);
//	Mat dst_img = Mat(ResImgSiz, src_img.type());
//	resize(src_img, dst_img, ResImgSiz, INTER_LINEAR);
//
//	Mat gray;
//	cvtColor(dst_img, gray, CV_BGR2GRAY);
//	Canny(gray, gray, 50, 250, 3);
//
//	namedWindow("PIC");
//	imshow("PIC", gray);
//	waitKey(0);
//}


/*函数功能：求两条直线交点*/
/*输入：两条Vec4i类型直线*/
/*返回：Point2f类型的点*/

Point2f getCrossPoint(Vec4i LineA, Vec4i LineB)
{
	double ka, kb;
	ka = (double)(LineA[3] - LineA[1]) / (double)(LineA[2] - LineA[0]); //求出LineA斜率
	kb = (double)(LineB[3] - LineB[1]) / (double)(LineB[2] - LineB[0]); //求出LineB斜率

	Point2f crossPoint;
	crossPoint.x = (ka*LineA[0] - LineA[1] - kb*LineB[0] + LineB[1]) / (ka - kb);
	crossPoint.y = (ka*kb*(LineA[0] - LineB[0]) + ka*LineB[1] - kb*LineA[1]) / (ka - kb);
	return crossPoint;
}


int main() {
	//fun();
	
	
	//【1】载入原始图和Mat变量定义   
	Mat src_img = imread("15.jpg");  //工程目录下应该有一张名为1.jpg的素材图
	//imshow("原始图", src_img);
	//cout << src_img.size().width;
	//waitKey(10000);
	double a = src_img.size().height;
	double b = src_img.size().width;
	double c = a / b;
	if (c > 1) {
		Size ResImgSiz = Size(330, 330*c);
		//Mat srcImage1 = Mat(ResImgSiz, src_img.type());
		resize(src_img, src_img, ResImgSiz, INTER_LINEAR);
		imshow("原始图", src_img);
		//waitKey(1000);

		IplImage imgTmp = src_img;
		IplImage *input = cvCloneImage(&imgTmp);
		cout << src_img.size().height;
		cout << src_img.size().width;
		cvSetImageROI(input, cvRect(0, 120, 330, 230));
		
		src_img = cv::cvarrToMat(input);
		//imshow("m1", src_img);
		//waitKey(10000);
	}
		//使用resize()函数统一图像大小
	Size ResImgSiz = Size(330, 230);
	Mat srcImage = Mat(ResImgSiz, src_img.type());
	resize(src_img, srcImage, ResImgSiz, INTER_LINEAR);
	cout << srcImage.size().height;
	Mat midImage, dstImage, outimg, lunk;//临时变量和目标图的定义

						   //【2】进行边缘检测和转化为灰度图
    //blur(srcImage, srcImage, Size(3, 3));
	//medianBlur(srcImage, srcImage, 3);
	cvtColor(srcImage, dstImage, CV_BGR2GRAY);//转化边缘检测后的图为灰度图
	//imshow("srcImage", dstImage);
	
	GaussianBlur(dstImage, dstImage, Size(3, 3), 0, 0);
//	imshow("srcImage2", dstImage);
	
	Canny(dstImage, midImage, 50, 100, 3);//进行一此canny边缘检测
	//imshow("mid", midImage);
	//Mat element = getStructuringElement(MORPH_RECT, Size(45, 5));
	//Mat element2 = getStructuringElement(MORPH_RECT, Size(55, 11));
	//morphologyEx(midImage, midImage, cv::MORPH_CLOSE, element);
	//morphologyEx(midImage, midImage, cv::MORPH_OPEN, element2);

	//imshow("mids", midImage);
	//waitKey(10000);
	//imshow("srcImage3", midImage);
	
   // cvtColor(midImage, dstImage, CV_GRAY2BGR);//转化边缘检测后的图为灰度图
	//cv::blur(dstImage, dstImage, cv::Size(3, 3));
	//threshold(midImage, dstImage, 0, 255, CV_THRESH_OTSU + CV_THRESH_BINARY);


	//vector<vector<Point> > contours;
	//vector<Vec4i>hierarchy;
	//cv::findContours(dstImage, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE); //找轮廓 
	//vector<Rect> rect;

	////绘制轮廓图
	//lunk = Mat::zeros(dstImage.size(), CV_8UC3);
	//for (int i = 0; i < hierarchy.size(); i++)
	//{
	//	Scalar color = Scalar(rand() % 255, rand() % 255, rand() % 255);
	//	drawContours(lunk, contours, i, color, CV_FILLED, 8, hierarchy);
	//}
	//imshow("轮廓图", lunk);


	//for (int i = 0; i < contours.size(); ++i)
	//{
	//	cout << contours[i].size() << endl;

	//	if (contours[i].size() > 500)//将比较小的轮廓剔除掉  
	//	{
	//		rect.push_back(boundingRect(contours[i]));
	//	}
	//}
	//
	//cout << rect.size() << endl;
	//rect.copyTo(outimg);
	//for (int j = 0; j < rect.size(); j++)
	//{
	//	float r = (float)rect[j].width / (float)rect[j].height;
	//	if (r < 1) {
	//		r = (float)rect[j].height / (float)rect[j].width;
	//	}
	//	cout << r << ",,," << rect[j].width << ",,," << rect[j].height << endl;
	//	if (r <= CARDRATIO_MAX && r >= CARDRATIO_MIN) {
	//		original(rect[j]).copyTo(outimg); // copy the region rect1 from the image to roi1
	//										  imshow("re", outimg);
	//	}
	//}
	


										  //【3】进行霍夫线变换

	//image为输入图像，要求是8位单通道图像
		//lines为输出的直线向量，每条线用4个元素表示，即直线的两个端点的4个坐标值
		//rho和theta分别为距离和角度的分辨率
		//threshold为阈值，即步骤3中的阈值
		//minLineLength为最小直线长度，在步骤5中要用到，即如果小于该值，则不被认为是一条直线
		//maxLineGap为最大直线间隙，在步骤4中要用到，即如果有两条线段是在一条直线上，但它们之间因为有间隙，所以被认为是两个线段，如果这个间隙大于该值，则被认为是两条线段，否则是一条。

	vector<Vec4i> lines;//定义一个矢量结构lines用于存放得到的线段矢量集合
	//HoughLinesP(midImage, lines, 1, CV_PI / 180, 70, 30, 10);
	HoughLinesP(midImage, lines, 1, CV_PI / 180, 70, 30, 10);
	//cout<<"线段个数"<< lines.size();




	//needed for visualization only//这里是将检测的线调整到延长至全屏，即射线的效果，其实可以不必这么做
	//for (unsigned int i = 0; i<lines.size(); i++)
	//{
	//	cv::Vec4i v = lines[i];
	//	lines[i][0] = 0;
	//	lines[i][1] = ((float)v[1] - v[3]) / (v[0] - v[2])* -v[0] + v[1];
	//	lines[i][2] = srcImage.cols;
	//	lines[i][3] = ((float)v[1] - v[3]) / (v[0] - v[2])*(srcImage.cols - v[2]) + v[3];
	//}



	//【4】依次在图中绘制出每条线段
	/*for (size_t i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(dstImage, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(186, 88, 255), 1, CV_AA);
	}*/
	//imshow("dstImage", dstImage);
	//waitKey(10000);


	std::vector<cv::Point2f> corners;//线的交点存储
	for (unsigned int i = 0; i<lines.size(); i++)
	{
		for (unsigned int j = i + 1; j<lines.size(); j++)
		{
			//float p = 0.0, q = 330.0, r = 230.0;
			cv::Point2f pt = getCrossPoint(lines[i], lines[j]);
			if((pt.x>0 && pt.y>0)&&(pt.x<330&&pt.y<230))
			{
				corners.push_back(pt);
				//cout << "ptttttttttt" << pt;
			}
		}
	}




	
	//【4】依次在图中绘制出交点
	for (size_t i = 0; i < corners.size(); i++)
	{
		circle(dstImage, corners[i], 3, CV_RGB(0, 255, 0), 2);


	}
	cout << "交点个数" << corners.size();
	imshow("dstImage", dstImage);
	//waitKey(10000);
	//std::vector<cv::Point2f> approx;
	//cv::approxPolyDP(cv::Mat(corners), approx, cv::arcLength(cv::Mat(corners), true)*0.02, true);

	//if (approx.size() != 4)
	//{
	//	std::cout << "The object is not quadrilateral（四边形）!" << std::endl;
	//	system("pause");
	//	//return -1;
	//}
	//get mass center
	for (unsigned int i = 0; i < corners.size(); i++)
	{
		center += corners[i];
	}
	center *= (1./ corners.size());
	sortCorners(corners, center);
	cv::Mat dst = srcImage.clone();
	//Draw Lines
	for (unsigned int i = 0; i<lines.size(); i++)
	{
		cv::Vec4i v = lines[i];
		cv::line(dst, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), CV_RGB(0, 255, 0));	//目标版块画绿线 
	}
	   // cv::Vec4i v = lines[0];
		//cv::line(dst, cv::Point(v[0], v[1]), cv::Point(v[2], v[3]), CV_RGB(0, 255, 255));	//目标版块画绿线 
		//cv::Vec4i q = lines[1];
		//cv::line(dst, cv::Point(q[0], q[1]), cv::Point(q[2], q[3]), CV_RGB(255, 255, 0));	//目标版块画绿线 
	//draw corner points
	cv::circle(dst, corners[0], 3, CV_RGB(255, 0, 0), 2);
	cv::circle(dst, corners[1], 3, CV_RGB(0, 255, 0), 2);
	cv::circle(dst, corners[2], 3, CV_RGB(0, 0, 255), 2);
	cv::circle(dst, corners[3], 3, CV_RGB(255, 255, 255), 2);
	//cv::circle(dst, cv::Point(1,1), 3, CV_RGB(255, 255, 0), 2);
	//draw mass center
	cout << "坐标" << corners[0]<<corners[1]<< corners[2]<<corners[3];
	cv::circle(dst, center, 3, CV_RGB(255, 255, 0), 2);
	imshow("dest", dst);
	//waitKey(10000);
	Mat quad= srcImage.clone();
	//cout << "dddddddddddd";
	double f = getDistance(corners[0], corners[1]);
	double g = getDistance(corners[0], corners[2]);
	double h = getDistance(corners[0], corners[3]);
	double j = getDistance(corners[1], corners[3]);
	double k = getDistance(corners[1], corners[2]);
	double l = getDistance(corners[2], corners[3]);
	cout << "g" << g << "f" << f << "h" << h;
	if(g>f&&g>h&&g>100&&f>100&&h>100&&j>k&&j>l&&j>100&&k>100&&l>100){
	cv::Mat quads = cv::Mat::zeros(230,330, CV_8UC3);//设定校正过的图片从320*240变为300*220
													 //corners of the destination image
	std::vector<cv::Point2f> quad_pts;
	quad_pts.push_back(cv::Point2f(0, 0));
	quad_pts.push_back(cv::Point2f(quads.cols, 0));//(220,0)
	quad_pts.push_back(cv::Point2f(quads.cols, quads.rows));//(220,300)
	quad_pts.push_back(cv::Point2f(0, quads.rows));

	// Get transformation matrix
	cv::Mat transmtx = cv::getPerspectiveTransform(corners, quad_pts);	//求源坐标系（已畸变的）与目标坐标系的转换矩阵

																		// Apply perspective transformation透视转换
	cv::warpPerspective(srcImage, quads, transmtx, quads.size());
	imshow("quad", quads);
	quad = quads.clone();
	}
	/*
    //Mat quad = imread("xxx.jpg");
   // imshow("quad", quad);
	IplImage imgTmp = quad;
	IplImage *input = cvCloneImage(&imgTmp);

	//cvSetImageROI(input, cvRect(20, 130, 400, 40));
	cvSetImageROI(input, cvRect(20, 120, 300, 40));
	Mat m1 = cv::cvarrToMat(input);
	imshow("m1", m1);
	*/

	//定位卡号区域
	
	//quad:矫正后的图像
	Mat cay;
	Canny(quad, cay, 100, 150, 3);//进行一此canny边缘检测
	imshow("cay", cay);
   // 使用开运算和闭运算让图像边缘成为一个整体
	//kernel = np.ones((10, 10), np.uint8)
		Mat element = getStructuringElement(MORPH_RECT, Size(25, 5));
		Mat element2 = getStructuringElement(MORPH_RECT, Size(75, 11));
		morphologyEx(cay, cay, cv::MORPH_CLOSE, element);
		morphologyEx(cay, cay, cv::MORPH_OPEN, element2);

		imshow("cayss", cay);
		//waitKey(10000);


 //矩形轮廓查找与筛选：
		Mat contour_image;
		//深拷贝  查找轮廓会改变源图像信息，需要重新 拷贝 图像
		contour_image = cay.clone();
		vector<vector<Point>> contours;
		findContours(contour_image, contours, CV_RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
		//画出轮廓
		drawContours(contour_image, contours, -1, Scalar(255), 1);
		imshow("testy", contour_image);
	//	waitKey(10000);


		//轮廓表示为一个矩形  卡号区域提取
		Mat roi_image;
		
		vector<Point> rectPoint;
		for (int i = 0; i < contours.size(); i++) {
			Rect r = boundingRect(Mat(contours[i]));
			//RotatedRect r = minAreaRect(Mat(contours[i]));
			cout << "contours " << i << "  height = " << r.height << "  width = " << r.width << "rate = " << ((float)r.width / r.height) <<"面积"<< (float)r.width* r.height << endl;
			if ((float)r.width / r.height >= 12 && (float)r.width / r.height <= 15 && ((float)r.width* r.height >=3000)) {
				cout << "r.x = " << r.x << "  r.y  = " << r.y << endl;
				rectangle(contour_image, r, Scalar(0, 0, 255), 2);
				imshow("contour_image.jpg", contour_image);
			//	waitKey(10000);
				Point p1, p2, p3, p4;
				p1.x = r.x;
				p1.y = r.y;
				p2.x = r.x + r.width;
				p2.x = r.y;
				p3.x = r.x + r.width;
				p3.y = r.y + r.height;
				p4.x = r.x;
				p4.y = r.y + r.height;


				rectPoint.push_back(p1);
				rectPoint.push_back(p2);
				rectPoint.push_back(p3);
				rectPoint.push_back(p4);


				for (int j = 0; j < contours[i].size(); j++) {
					cout << "point = " << contours[i][j] << endl;
				}
				//rectangle(image, r, Scalar(0, 0, 255), 3);
				roi_image = quad(r);
			}
		}
		if (roi_image.empty()) {
			IplImage imgTmp = quad;
			IplImage *input = cvCloneImage(&imgTmp);

			//cvSetImageROI(input, cvRect(20, 130, 400, 40));
			cvSetImageROI(input, cvRect(20, 120, 300, 40));
			roi_image = cv::cvarrToMat(input);
		}
		
	//	imwrite("roi_image.jpg", roi_image);
		//图片放大
		//Mat large_image, large_image2;
		int col = roi_image.cols, row = roi_image.rows;
		resize(roi_image, roi_image, Size(300, 300*row/col));
		imshow("test", roi_image);
	//	waitKey(10000);
		//imshow("lage", m1);
		//waitKey(5000);
	//	Canny(large_image, large_image, 50, 150, 3);//进行一此canny边缘检测

	
		
	Mat m2;
		cvtColor(roi_image, m2, COLOR_BGR2GRAY);
	for (int q = 0; q < m2.rows; q++) {
		for (int t = 0; t < m2.cols; t++) {
			if (m2.at<uchar>(q, t) >40) {
				m2.at<uchar>(q, t) = 0;
			}
			else {
				m2.at<uchar>(q, t) = 255;
			}
			
		}
	}
	
	double count = 0;
	for (int q = 0; q < m2.rows; q++) {
		for (int t = 0; t < m2.cols; t++) {
			if (m2.at<uchar>(q, t) ==255) {
				count = count + 1;
			}
		}
	}
	double all = m2.rows*m2.cols;
	double jieguo = all / count;
	cout << "白色像素点个数" << count << "count/all" <<jieguo;
	Mat ycc,m1_canny;
	if (count < 500) {
		for (int x = 0; x < roi_image.rows; x++)
		{
			for (int y = 0; y < roi_image.cols; y++)
			{
				for (int c = 0; c < 3; c++)
				{
					roi_image.at<Vec3b>(x, y)[c] = saturate_cast<uchar>(1.5*(roi_image.at<Vec3b>(x, y)[c]) + 0);
				}
			}
		}
		imshow("m1c", roi_image);
		Canny(roi_image, m1_canny, 50, 150, 3);
		imshow("m1_canny", m1_canny);
		cvtColor(roi_image, ycc, CV_RGB2YCrCb);
		imshow("ycc", ycc);

		vector<Mat> channels;  //定义Mat向量容器保存拆分后的数据
		Mat YChannel;
		Mat UChannel;
		Mat VChannel;
		Mat abs;
		split(ycc, channels);
		YChannel = channels.at(0);
		UChannel = channels.at(1);
		VChannel = channels.at(2);
		imshow("VChannels", VChannel);
		Canny(YChannel, YChannel, 70, 210, 3);
		Canny(UChannel, UChannel, 50, 60, 3);
		Canny(VChannel, VChannel, 50, 100, 3);
		imshow("YChannel", YChannel);
		imshow("UChannel", UChannel);
		imshow("VChannel", VChannel);
		absdiff(m1_canny, VChannel, abs);
		imshow("abs", abs);
		m2 = YChannel.clone();
	
	}
	
	//imshow("large_image", ycc);
	
		//将最左右边一列变为黑色
		for (int e = 0; e < m2.rows; e++) {
			m2.at<uchar>(e, 0) = 0;
			m2.at<uchar>(e, m2.cols-1) = 0;
		}
		imshow("eeee", m2);
			//统计原图片中每列白色像素数目
			//blur(m2, m2, Size(3, 3));//模糊，去锯齿
			int src_width = m2.cols;
			int src_height = m2.rows;
			int* projectValArry = new int[src_width]();//创建用于储存每列白色像素个数的数组
													   //memset(projectValArry, 0, src_width*4);//初始化数组
													   //取列白色像素个数
			for (int i = 0; i < src_height; i++) {
				for (int j = 0; j < src_width; j++) {
					if (m2.at<uchar>(i, j)) {
						projectValArry[j]++;
					}
				}
			}
			//将每列白色像素数目绘制成直方图
			//定义画布 绘制垂直投影下每列白色像素的数目
			Mat verticalProjectionMat(src_height, src_width, CV_8UC1, Scalar(0));
			for (int i = 0; i< src_width; i++) {
				for (int j = 0; j < projectValArry[i]; j++) {
					verticalProjectionMat.at<uchar>(src_height - j - 1, i) = 255;
				}
			}
			imshow("verticalProjectionMat", verticalProjectionMat);

			//*********根据每列白色像素数目设置截取起始和截止列
			//定义Mat vector ，存储图片组
			vector<Mat> split_src;
			//定义标志，用来指示在白色像素区还是在全黑区域
			bool white_block = 0, black_block = 0;
			//定义列temp_col_forword  temp_col_behind，记录字符截取起始列和截止列
			int temp_col_forword = 0, temp_col_behind = 0;
			Mat split_temp;
			//遍历数组projectValArry
			
			for (int i = 0; i < src_width; i++) {
				if (projectValArry[i]) {//表示区域有白色像素
					white_block = 1;
					black_block = 0;
				}
				else {				//若无白色像素（进入黑色区域）
					if (white_block == 1) {//若前一列有白色像素
						temp_col_behind = i;//取当前列为截止列
											//截取下一部分
						
						split_temp= roi_image(Rect(temp_col_forword, 0, temp_col_behind - temp_col_forword, src_height)).clone();
						//split_src.push_back(split_temp);
						//细分割
						//Mat split2 = split_temp.clone();
						Mat split2;
						//split_temp.copyTo(split2);
						
						if(split_temp.size().width>20){
							
							for (int z=0; z < split_temp.size().width; z =z+14) {
								//cout << "split_temp.size().width" << split_temp.size().width << endl;
								//cout << "split2.size().width" << split2.size().width << endl;
								//cout << "z:" << z << endl;
								if (split_temp.size().width - z > 14) {
									
									split2= split_temp(Rect(z, 0, 14, src_height)).clone();
									split_src.push_back(split2);
								}
							}
						}
						if (split_temp.size().width>4&& split_temp.size().width<20){
						split_src.push_back(split_temp);
						}
						
					}
					temp_col_forword = i;//记录最新黑色区域的列号，记为起始列
					black_block = 1;//表示进入黑色区域
					white_block = 0;
				}
			}

			for (int i = 0; i < split_src.size(); i++) {
			char window[20];
			sprintf(window, " split: %d", i);
			imshow(window, split_src[i]);
		//	string Img_Name = "C:\\Users\\wangshuang\\Desktop\\Car\\image\\img2\\" + to_string(i) + ".bmp";
		//	imwrite(Img_Name, split_src[i]);

			}
			waitKey(0);
			


	

		
			//return input_src;
		///////////////////////////////////////
		

	//	imshow("test", m2);
		waitKey();

}

