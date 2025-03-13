import { useState } from "react";
import { Form } from "@/components/ui/form";
import { zodResolver } from "@hookform/resolvers/zod";
import { useForm } from "react-hook-form";
import { Input } from "@/components/ui/input";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardDescription, CardFooter, CardHeader, CardTitle } from "@/components/ui/card";
import { Link, useLocation } from "wouter";
import { loginSchema, type LoginUser } from "@shared/schema";
import { useToast } from "@/hooks/use-toast";
import { apiRequest } from "@/lib/queryClient";
import { useAuth } from "@/context/auth-context";
import { Loader2 } from "lucide-react";

export default function Login() {
  const [, setLocation] = useLocation();
  const { toast } = useToast();
  const { login } = useAuth();
  const [isLoading, setIsLoading] = useState(false);

  const form = useForm<LoginUser>({
    resolver: zodResolver(loginSchema),
    defaultValues: {
      username: "",
      password: "",
    },
  });

  const onSubmit = async (values: LoginUser) => {
    setIsLoading(true);
    try {
      const response = await apiRequest('POST', '/api/auth/login', values);
      if (response.token) {
        login(response.token);
        toast({
          title: "Welcome back!",
          description: "Successfully logged in to PanicSense PH",
        });
        setLocation('/dashboard');
      }
    } catch (error) {
      toast({
        title: "Login failed",
        description: error instanceof Error ? error.message : "Please check your credentials",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen w-full flex flex-col items-center justify-center bg-gradient-to-br from-blue-50 to-indigo-50 p-4">
      <div className="w-full max-w-md space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-gray-900">
            PanicSense PH
          </h1>
          <p className="text-lg text-gray-600">
            Disaster Sentiment Analysis Platform
          </p>
        </div>

        <Card className="border-0 shadow-xl">
          <CardHeader className="space-y-1">
            <CardTitle className="text-2xl">Sign in to your account</CardTitle>
            <CardDescription>
              Enter your credentials to access the dashboard
            </CardDescription>
          </CardHeader>
          <CardContent>
            <Form {...form}>
              <form onSubmit={form.handleSubmit(onSubmit)} className="space-y-4">
                <div className="space-y-2">
                  <Input
                    {...form.register("username")}
                    placeholder="Username"
                    type="text"
                    autoComplete="username"
                    className="h-11"
                  />
                  {form.formState.errors.username && (
                    <p className="text-sm text-red-500">
                      {form.formState.errors.username.message}
                    </p>
                  )}
                </div>

                <div className="space-y-2">
                  <Input
                    {...form.register("password")}
                    placeholder="Password"
                    type="password"
                    autoComplete="current-password"
                    className="h-11"
                  />
                  {form.formState.errors.password && (
                    <p className="text-sm text-red-500">
                      {form.formState.errors.password.message}
                    </p>
                  )}
                </div>

                <Button
                  type="submit"
                  className="w-full h-11 text-lg font-semibold bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 transition-all duration-300"
                  disabled={isLoading}
                >
                  {isLoading ? (
                    <Loader2 className="h-5 w-5 animate-spin" />
                  ) : (
                    "Sign in"
                  )}
                </Button>
              </form>
            </Form>
          </CardContent>
          <CardFooter className="flex flex-col space-y-4 border-t pt-6">
            <p className="text-sm text-slate-600 text-center">
              Don't have an account?{" "}
              <Link 
                href="/signup" 
                className="font-semibold text-blue-600 hover:text-blue-800 transition-colors"
              >
                Sign up
              </Link>
            </p>
          </CardFooter>
        </Card>
      </div>
    </div>
  );
}