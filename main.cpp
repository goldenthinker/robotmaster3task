#include <iostream>
#include <vector>
#include <string>
#include <opencv2/opencv.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;
using namespace cv;

// 弹道模型结构体
struct TrajectoryModel {
    double x0, y0;      // 初始位置
    double vx0, vy0;    // 初始速度 (px/s)
    double g;           // 重力加速度 (px/s²)
    double k;           // 阻力系数 (1/s)
    
    TrajectoryModel(double x0_, double y0_, double vx0_, double vy0_, double g_, double k_)
        : x0(x0_), y0(y0_), vx0(vx0_), vy0(vy0_), g(g_), k(k_) {}
};

// Ceres代价函数
struct TrajectoryCostFunction {
    double t;    // 时间 (s)
    double x;    // 观测x坐标
    double y;    // 观测y坐标
    double fps;  // 视频帧率
    
    TrajectoryCostFunction(double time, double x_obs, double y_obs, double frame_rate) 
        : t(time), x(x_obs), y(y_obs), fps(frame_rate) {}
    
    template<typename T>
    bool operator()(const T* const params, T* residual) const {
        // 参数顺序: [vx0, vy0, g, k]
        const T& vx0 = params[0];
        const T& vy0 = params[1];
        const T& g = params[2];
        const T& k = params[3];
        
        T delta_t = T(t);
        
        // 弹道模型公式
        T exp_kt = ceres::exp(-k * delta_t);
        
        // x方向位置
        T x_pred = T(x0) + (vx0 / k) * (T(1.0) - exp_kt);
        
        // y方向位置
        T y_pred = T(y0) + ((vy0 + g / k) / k) * (T(1.0) - exp_kt) - (g / k) * delta_t;
        
        // 残差 (观测值 - 预测值)
        residual[0] = T(x) - x_pred;
        residual[1] = T(y) - y_pred;
        
        return true;
    }
    
private:
    // 假设初始位置已知或从第一帧获取
    double x0 = 0.0;  // 需要根据实际视频调整
    double y0 = 0.0;  // 需要根据实际视频调整
};

class TrajectoryFitter {
private:
    vector<double> time_points;
    vector<double> x_observations;
    vector<double> y_observations;
    double fps;
    double initial_x0, initial_y0;
    
public:
    TrajectoryFitter(double frame_rate) : fps(frame_rate) {}
    
    // 从视频中提取蓝色弹道轨迹点
    bool extractTrajectoryFromVideo(const string& video_path) {
        VideoCapture cap(video_path);
        if (!cap.isOpened()) {
            cerr << "无法打开视频文件: " << video_path << endl;
            return false;
        }
        
        double video_fps = cap.get(CAP_PROP_FPS);
        if (abs(video_fps - fps) > 1.0) {
            cout << "警告: 视频FPS与指定FPS不一致，使用视频FPS: " << video_fps << endl;
            fps = video_fps;
        }
        
        Mat frame, hsv, blue_mask;
        vector<Point2f> trajectory_points;
        
        // 蓝色范围 (HSV颜色空间)
        Scalar lower_blue = Scalar(100, 50, 50);   // 需要根据实际蓝色调整
        Scalar upper_blue = Scalar(140, 255, 255);
        
        int frame_count = 0;
        while (cap.read(frame)) {
            if (frame.empty()) break;
            
            // 转换到HSV颜色空间
            cvtColor(frame, hsv, COLOR_BGR2HSV);
            
            // 创建蓝色掩码
            inRange(hsv, lower_blue, upper_blue, blue_mask);
            
            // 寻找轮廓
            vector<vector<Point>> contours;
            findContours(blue_mask, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
            
            // 寻找最大轮廓 (假设为弹道)
            if (!contours.empty()) {
                auto largest_contour = max_element(contours.begin(), contours.end(),
                    [](const vector<Point>& a, const vector<Point>& b) {
                        return contourArea(a) < contourArea(b);
                    });
                
                // 计算轮廓中心
                Moments m = moments(*largest_contour);
                if (m.m00 > 0) {
                    double x = m.m10 / m.m00;
                    double y = m.m01 / m.m00;
                    trajectory_points.push_back(Point2f(x, y));
                }
            }
            
            frame_count++;
        }
        
        cap.release();
        
        if (trajectory_points.empty()) {
            cerr << "未检测到蓝色弹道轨迹" << endl;
            return false;
        }
        
        // 设置初始位置 (第一帧的位置)
        initial_x0 = trajectory_points[0].x;
        initial_y0 = trajectory_points[0].y;
        
        // 准备观测数据
        for (size_t i = 0; i < trajectory_points.size(); ++i) {
            double t = i / fps;  // 时间 (秒)
            time_points.push_back(t);
            x_observations.push_back(trajectory_points[i].x);
            y_observations.push_back(trajectory_points[i].y);
        }
        
        cout << "成功提取 " << trajectory_points.size() << " 个轨迹点" << endl;
        return true;
    }
    
    // 手动设置观测数据 (用于测试)
    void setObservationData(const vector<double>& times, 
                           const vector<double>& x_obs, 
                           const vector<double>& y_obs,
                           double x0, double y0) {
        time_points = times;
        x_observations = x_obs;
        y_observations = y_obs;
        initial_x0 = x0;
        initial_y0 = y0;
    }
    
    // 使用Ceres进行拟合
    bool fitTrajectory(double& vx0, double& vy0, double& g, double& k) {
        if (time_points.empty()) {
            cerr << "没有观测数据可供拟合" << endl;
            return false;
        }
        
        // 初始参数猜测
        double initial_params[4] = {vx0, vy0, g, k};
        
        ceres::Problem problem;
        
        // 添加残差块
        for (size_t i = 0; i < time_points.size(); ++i) {
            ceres::CostFunction* cost_function = 
                new ceres::AutoDiffCostFunction<TrajectoryCostFunction, 2, 4>(
                    new TrajectoryCostFunction(time_points[i], x_observations[i], 
                                             y_observations[i], fps));
            
            // 使用鲁棒核函数应对异常值
            ceres::LossFunction* loss = new ceres::HuberLoss(1.0);
            problem.AddResidualBlock(cost_function, loss, initial_params);
        }
        
        // 设置参数约束
        // g的范围: 100-1000 px/s²
        problem.SetParameterLowerBound(initial_params, 2, 100.0);
        problem.SetParameterUpperBound(initial_params, 2, 1000.0);
        
        // k的范围: 0.01-1 1/s
        problem.SetParameterLowerBound(initial_params, 3, 0.01);
        problem.SetParameterUpperBound(initial_params, 3, 1.0);
        
        // 配置求解器选项
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::DENSE_QR;  // 小规模问题
        options.minimizer_progress_to_stdout = true;
        options.max_num_iterations = 100;
        options.function_tolerance = 1e-6;
        
        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);
        
        cout << summary.FullReport() << endl;
        
        // 输出结果
        vx0 = initial_params[0];
        vy0 = initial_params[1];
        g = initial_params[2];
        k = initial_params[3];
        
        // 计算拟合误差
        double total_error = 0.0;
        for (size_t i = 0; i < time_points.size(); ++i) {
            double t = time_points[i];
            double exp_kt = exp(-k * t);
            
            double x_pred = initial_x0 + (vx0 / k) * (1.0 - exp_kt);
            double y_pred = initial_y0 + ((vy0 + g / k) / k) * (1.0 - exp_kt) - (g / k) * t;
            
            double error = sqrt(pow(x_observations[i] - x_pred, 2) + 
                              pow(y_observations[i] - y_pred, 2));
            total_error += error;
        }
        
        double avg_error = total_error / time_points.size();
        cout << "平均拟合误差: " << avg_error << " 像素" << endl;
        
        return summary.IsSolutionUsable();
    }
    
    // 获取初始位置
    pair<double, double> getInitialPosition() const {
        return {initial_x0, initial_y0};
    }
};

int main() {
    // 参数设置
    double fps = 60.0;
    string video_path = "/home/zhanruixuan/Downloads/video.mp4";  // 替换为实际视频路径
    
    // 创建拟合器
    TrajectoryFitter fitter(fps);
    
    // 从视频提取轨迹
    if (!fitter.extractTrajectoryFromVideo(video_path)) {
        // 如果视频处理失败，使用模拟数据进行测试
        cout << "使用模拟数据进行测试..." << endl;
        
        vector<double> times, x_obs, y_obs;
        double x0 = 100.0, y0 = 500.0;
        double vx0 = 200.0, vy0 = -300.0, g = 500.0, k = 0.1;
        
        // 生成模拟轨迹
        for (int i = 0; i < 100; ++i) {
            double t = i / fps;
            times.push_back(t);
            
            double exp_kt = exp(-k * t);
            double x = x0 + (vx0 / k) * (1.0 - exp_kt);
            double y = y0 + ((vy0 + g / k) / k) * (1.0 - exp_kt) - (g / k) * t;
            
            // 添加一些噪声
            x_obs.push_back(x + (rand() % 10 - 5));
            y_obs.push_back(y + (rand() % 10 - 5));
        }
        
        fitter.setObservationData(times, x_obs, y_obs, x0, y0);
    }
    
    // 初始参数猜测 (需要根据实际情况调整)
    double vx0 = 150.0, vy0 = -200.0, g = 300.0, k = 0.05;
    
    // 进行拟合
    if (fitter.fitTrajectory(vx0, vy0, g, k)) {
        cout << "\n=== 拟合结果 ===" << endl;
        cout << "初始速度 vx0: " << vx0 << " px/s" << endl;
        cout << "初始速度 vy0: " << vy0 << " px/s" << endl;
        cout << "重力加速度 g: " << g << " px/s²" << endl;
        cout << "阻力系数 k: " << k << " 1/s" << endl;
        
        auto [x0, y0] = fitter.getInitialPosition();
        cout << "初始位置 x0: " << x0 << ", y0: " << y0 << endl;
        
        // 计算合速度
        double v0 = sqrt(vx0 * vx0 + vy0 * vy0);
        cout << "合速度: " << v0 << " px/s" << endl;
    } else {
        cerr << "拟合失败" << endl;
        return -1;
    }
    
    return 0;
}
